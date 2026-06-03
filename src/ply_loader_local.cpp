#include "ply_loader_local.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace app {
namespace {

/// Sigmoid：将 opacity_logit 映射到 [0, 1]。
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// 移除行尾 CR/LF，兼容不同平台换行。
void stripCrLf(std::string& line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
}

/// PLY 顶点属性描述。
struct PlyProperty {
    std::string name;
    std::string type;
};

/// PLY 头部解析结果。
struct PlyHeader {
    bool binary = false;
    bool little_endian = true;
    size_t vertex_count = 0;
    std::vector<PlyProperty> properties;
};

/// 解析 PLY 头（格式、顶点数、属性列表）。
bool parseHeader(std::istream& in, PlyHeader& hdr) {
    std::string line;
    if (!std::getline(in, line)) return false;
    stripCrLf(line);
    if (line != "ply") return false;
    bool in_vertex = false;
    while (std::getline(in, line)) {
        stripCrLf(line);
        if (line == "end_header") break;
        std::istringstream ss(line);
        std::string token;
        ss >> token;
        if (token == "format") {
            std::string fmt, version;
            ss >> fmt >> version;
            if (fmt == "binary_little_endian") {
                hdr.binary = true;
                hdr.little_endian = true;
            } else if (fmt == "ascii") {
                hdr.binary = false;
            } else {
                return false;
            }
        } else if (token == "element") {
            std::string elem;
            size_t count = 0;
            ss >> elem >> count;
            in_vertex = (elem == "vertex");
            if (in_vertex) hdr.vertex_count = count;
        } else if (token == "property" && in_vertex) {
            PlyProperty prop;
            ss >> prop.type >> prop.name;
            hdr.properties.push_back(prop);
        }
    }
    return hdr.vertex_count > 0 && !hdr.properties.empty();
}

/// 查询指定属性在顶点行中的列索引。
int propertyIndex(const PlyHeader& hdr, const std::string& name) {
    for (size_t i = 0; i < hdr.properties.size(); ++i) {
        if (hdr.properties[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

/// 返回 PLY 属性类型的字节宽度。
size_t propertyByteSize(const std::string& type) {
    if (type == "float" || type == "float32") return 4;
    if (type == "double") return 8;
    if (type == "uchar" || type == "uint8") return 1;
    if (type == "int" || type == "int32") return 4;
    return 4;
}

/// 计算单个顶点记录的总字节跨度。
size_t vertexStride(const PlyHeader& hdr) {
    size_t stride = 0;
    for (const auto& p : hdr.properties) stride += propertyByteSize(p.type);
    return stride;
}

/// 读取 little-endian 浮点数。
float readFloatLE(const uint8_t* data) {
    uint32_t bits;
    std::memcpy(&bits, data, 4);
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

}  // namespace

/// 从 3DGS 风格 PLY 读取高斯数组（支持降采样与 reservoir）。
bool loadGaussianPlyLocal(const std::string& path, std::vector<gsplat::Gaussian>& out,
                          const PlyLoadOptions& opts) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    PlyHeader hdr;
    if (!parseHeader(file, hdr)) return false;

    const int ix = propertyIndex(hdr, "x");
    const int iy = propertyIndex(hdr, "y");
    const int iz = propertyIndex(hdr, "z");
    if (ix < 0 || iy < 0 || iz < 0) return false;

    const int i_dc0 = propertyIndex(hdr, "f_dc_0");
    const int i_dc1 = propertyIndex(hdr, "f_dc_1");
    const int i_dc2 = propertyIndex(hdr, "f_dc_2");
    const bool has_sh = (i_dc0 >= 0 && i_dc1 >= 0 && i_dc2 >= 0);

    std::vector<int> i_rest(45, -1);
    int rest_count = 0;
    for (int k = 0; k < 45; ++k) {
        i_rest[static_cast<size_t>(k)] = propertyIndex(hdr, "f_rest_" + std::to_string(k));
        if (i_rest[static_cast<size_t>(k)] >= 0) ++rest_count;
    }
    const bool has_sh_rest = (rest_count >= 45);

    const int i_opacity = propertyIndex(hdr, "opacity");
    const int i_s0 = propertyIndex(hdr, "scale_0");
    const int i_s1 = propertyIndex(hdr, "scale_1");
    const int i_s2 = propertyIndex(hdr, "scale_2");
    const int i_r0 = propertyIndex(hdr, "rot_0");
    const int i_r1 = propertyIndex(hdr, "rot_1");
    const int i_r2 = propertyIndex(hdr, "rot_2");
    const int i_r3 = propertyIndex(hdr, "rot_3");

    const size_t stride = vertexStride(hdr);
    std::vector<size_t> offsets(hdr.properties.size(), 0);
    {
        size_t off = 0;
        for (size_t i = 0; i < hdr.properties.size(); ++i) {
            offsets[i] = off;
            off += propertyByteSize(hdr.properties[i].type);
        }
    }

    const bool reservoir =
        opts.max_points > 0 && hdr.vertex_count > opts.max_points && !opts.stride_subsample;
    size_t step = 1;
    if (!reservoir && opts.max_points > 0 && hdr.vertex_count > opts.max_points) {
        step = hdr.vertex_count / opts.max_points;
        if (step < 1) step = 1;
    }
    const size_t target_count =
        reservoir ? opts.max_points
                  : (opts.max_points > 0 ? std::min(hdr.vertex_count, opts.max_points) : hdr.vertex_count);

    out.clear();
    out.reserve(target_count);

    size_t seen = 0;
    std::mt19937 rng;
    if (reservoir) rng.seed(std::random_device{}());

    auto pushGaussian = [&](const gsplat::Gaussian& in) {
        if (sigmoid(in.opacity_logit) < opts.min_opacity) return;
        if (!reservoir) {
            out.push_back(in);
            return;
        }
        ++seen;
        if (out.size() < target_count) {
            out.push_back(in);
        } else {
            std::uniform_int_distribution<size_t> dist(0, seen - 1);
            const size_t j = dist(rng);
            if (j < target_count) out[j] = in;
        }
    };

    auto fillFromRow = [&](const auto& getF) {
        gsplat::Gaussian g;
        g.x = getF(ix);
        g.y = getF(iy);
        g.z = getF(iz);
        if (i_opacity >= 0) g.opacity_logit = getF(i_opacity);
        if (i_s0 >= 0) {
            g.scale_log[0] = getF(i_s0);
            g.scale_log[1] = getF(i_s1);
            g.scale_log[2] = getF(i_s2);
        }
        if (i_r0 >= 0) {
            g.rot[0] = getF(i_r0);
            g.rot[1] = getF(i_r1);
            g.rot[2] = getF(i_r2);
            g.rot[3] = getF(i_r3);
        }
        if (has_sh) {
            g.sh[0] = getF(i_dc0);
            g.sh[1] = getF(i_dc1);
            g.sh[2] = getF(i_dc2);
            const bool load_rest = has_sh_rest && !opts.drop_sh_rest;
            g.has_sh_rest = load_rest;
            if (load_rest) {
                for (int k = 0; k < 45; ++k) g.sh[3 + k] = getF(i_rest[static_cast<size_t>(k)]);
            }
        }
        pushGaussian(g);
    };

    const size_t progress_step = hdr.vertex_count > 200000 ? hdr.vertex_count / 20 : 0;

    if (hdr.binary) {
        if (!hdr.little_endian) return false;
        std::vector<uint8_t> row(stride);
        for (size_t i = 0; i < hdr.vertex_count; ++i) {
            if (progress_step > 0 && i > 0 && (i % progress_step) == 0) {
                const int pct =
                    static_cast<int>((100.0 * static_cast<double>(i)) / static_cast<double>(hdr.vertex_count));
                std::cerr << "  PLY scan ~" << pct << "% (" << out.size() << " kept)\n";
            }
            if (!reservoir && step > 1 && (i % step) != 0) {
                file.seekg(static_cast<std::streamoff>(stride), std::ios::cur);
                if (!file) break;
                continue;
            }
            file.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(stride));
            if (!file) break;
            auto getF = [&](int idx) -> float {
                if (idx < 0) return 0.f;
                return readFloatLE(row.data() + offsets[static_cast<size_t>(idx)]);
            };
            fillFromRow(getF);
            if (!reservoir && out.size() >= target_count) break;
        }
    } else {
        for (size_t i = 0; i < hdr.vertex_count; ++i) {
            std::vector<float> values(hdr.properties.size());
            for (size_t p = 0; p < hdr.properties.size(); ++p) file >> values[p];
            if (!file) break;
            if (step > 1 && (i % step) != 0) continue;
            auto getF = [&](int idx) -> float {
                return (idx >= 0 && static_cast<size_t>(idx) < values.size()) ? values[static_cast<size_t>(idx)]
                                                                               : 0.f;
            };
            fillFromRow(getF);
            if (!reservoir && out.size() >= target_count) break;
        }
    }
    return !out.empty();
}

}  // namespace app
