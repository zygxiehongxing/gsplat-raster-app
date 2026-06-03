#include "camera_truth.h"

#include <cmath>
#include <fstream>
#include <vector>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

/// 以 JSON 数组格式写入 double 数组字段。
void writeArray(std::ostream& out, const char* key, const double* v, int n) {
    out << "  \"" << key << "\": [";
    for (int i = 0; i < n; ++i) {
        if (i) out << ", ";
        out << std::setprecision(9) << v[i];
    }
    out << "],\n";
}

/// 以 JSON 数组格式写入 float 数组字段。
void writeArrayF(std::ostream& out, const char* key, const float* v, int n) {
    out << "  \"" << key << "\": [";
    for (int i = 0; i < n; ++i) {
        if (i) out << ", ";
        out << std::setprecision(9) << v[i];
    }
    out << "],\n";
}

/// 从 JSON 文本中按 key 读取固定长度数组。
bool readArray(const std::string& json, const char* key, double* out, int n) {
    const std::string token = std::string("\"") + key + "\"";
    const size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return false;
    const size_t lb = json.find('[', key_pos);
    if (lb == std::string::npos) return false;
    std::vector<double> vals;
    int depth = 0;
    for (size_t i = lb; i < json.size(); ++i) {
        if (json[i] == '[') ++depth;
        else if (json[i] == ']') {
            --depth;
            if (depth == 0) {
                std::string chunk = json.substr(lb + 1, i - lb - 1);
                size_t start = 0;
                while (start < chunk.size()) {
                    while (start < chunk.size() &&
                           (chunk[start] == ' ' || chunk[start] == '\t' || chunk[start] == '\n' ||
                            chunk[start] == '\r')) {
                        ++start;
                    }
                    if (start >= chunk.size()) break;
                    size_t end = start;
                    while (end < chunk.size() && chunk[end] != ',') ++end;
                    vals.push_back(std::stod(chunk.substr(start, end - start)));
                    start = end + 1;
                }
                break;
            }
        }
    }
    if (static_cast<int>(vals.size()) < n) return false;
    for (int i = 0; i < n; ++i) out[i] = vals[static_cast<size_t>(i)];
    return true;
}

/// 从 JSON 文本中按 key 读取整型字段。
bool readInt(const std::string& json, const char* key, int& out) {
    const std::string token = std::string("\"") + key + "\"";
    const size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return false;
    const size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) return false;
    out = std::stoi(json.substr(colon + 1));
    return true;
}

/// 从 JSON 文本中按 key 读取浮点字段。
bool readFloat(const std::string& json, const char* key, float& out) {
    const std::string token = std::string("\"") + key + "\"";
    const size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return false;
    const size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) return false;
    out = std::stof(json.substr(colon + 1));
    return true;
}

/// 从 JSON 文本中按 key 读取布尔字段。
bool readBool(const std::string& json, const char* key, bool& out) {
    const std::string token = std::string("\"") + key + "\"";
    const size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return false;
    const size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) return false;
    const std::string tail = json.substr(colon + 1);
    if (tail.find("true") != std::string::npos) {
        out = true;
        return true;
    }
    if (tail.find("false") != std::string::npos) {
        out = false;
        return true;
    }
    return false;
}

/// 读取整个文本文件。
std::string readAllText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

/// 序列化并保存 CameraTruth 到 JSON。
bool saveCameraTruth(const std::string& path, const CameraTruth& truth) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"width\": " << truth.width << ",\n";
    out << "  \"height\": " << truth.height << ",\n";
    out << "  \"mat_mode\": " << truth.mat_mode << ",\n";
    out << "  \"proj_mul_pv\": " << (truth.proj_mul_pv ? "true" : "false") << ",\n";
    out << "  \"scale_modifier\": " << truth.scale_modifier << ",\n";
    out << "  \"dc_only\": " << (truth.dc_only ? "true" : "false") << ",\n";
    out << "  \"drop_sh_rest\": " << (truth.drop_sh_rest ? "true" : "false") << ",\n";
    out << "  \"min_opacity\": " << truth.min_opacity << ",\n";
    out << "  \"visible\": " << truth.visible << ",\n";
    writeArray(out, "osg_view", truth.osg_view, 16);
    writeArray(out, "osg_proj", truth.osg_proj, 16);
    writeArrayF(out, "view", truth.sdk.view, 16);
    writeArrayF(out, "proj", truth.sdk.proj, 16);
    writeArrayF(out, "cam_pos", truth.sdk.cam_pos, 3);
    out << "  \"tan_fovx\": " << truth.sdk.tan_fovx << ",\n";
    out << "  \"tan_fovy\": " << truth.sdk.tan_fovy << "\n";
    out << "}\n";
    return true;
}

/// 反序列化并加载 CameraTruth。
bool loadCameraTruth(const std::string& path, CameraTruth& truth) {
    const std::string json = readAllText(path);
    if (json.empty()) return false;
    if (!readInt(json, "width", truth.width)) return false;
    if (!readInt(json, "height", truth.height)) return false;
    readInt(json, "mat_mode", truth.mat_mode);
    readBool(json, "proj_mul_pv", truth.proj_mul_pv);
    readFloat(json, "scale_modifier", truth.scale_modifier);
    readBool(json, "dc_only", truth.dc_only);
    readBool(json, "drop_sh_rest", truth.drop_sh_rest);
    readFloat(json, "min_opacity", truth.min_opacity);
    readInt(json, "visible", truth.visible);
    if (!readArray(json, "osg_view", truth.osg_view, 16)) return false;
    if (!readArray(json, "osg_proj", truth.osg_proj, 16)) return false;
    double tmp[16];
    if (!readArray(json, "view", tmp, 16)) return false;
    for (int i = 0; i < 16; ++i) truth.sdk.view[i] = static_cast<float>(tmp[i]);
    if (!readArray(json, "proj", tmp, 16)) return false;
    for (int i = 0; i < 16; ++i) truth.sdk.proj[i] = static_cast<float>(tmp[i]);
    if (!readArray(json, "cam_pos", tmp, 3)) return false;
    for (int i = 0; i < 3; ++i) truth.sdk.cam_pos[i] = static_cast<float>(tmp[i]);
    if (!readFloat(json, "tan_fovx", truth.sdk.tan_fovx)) return false;
    if (!readFloat(json, "tan_fovy", truth.sdk.tan_fovy)) return false;
    return truth.sdk.tan_fovx > 0.f && truth.sdk.tan_fovy > 0.f;
}

/// 将真值中的渲染参数覆盖到 raster 当前设置。
void applyTruthRenderSettings(gsplat::Rasterizer& raster, const CameraTruth& truth) {
    gsplat::RenderSettings rs = raster.settings();
    rs.scale_modifier = truth.scale_modifier;
    rs.dc_only = truth.dc_only;
    raster.setSettings(rs);
    (void)truth.drop_sh_rest;
    (void)truth.min_opacity;
}

/// 输出真值摘要信息，便于对齐日志排查。
void printCameraTruthSummary(const CameraTruth& truth, const char* label) {
    std::cout << "[TRUTH] " << label << " " << truth.width << "x" << truth.height << " mat_mode=" << truth.mat_mode
              << " scale=" << truth.scale_modifier << " visible=" << truth.visible << " / "
              << "tan=(" << truth.sdk.tan_fovx << "," << truth.sdk.tan_fovy << ")\n";
}

/// 打印两矩阵的最大绝对差与对应索引。
void logMatrixDiff(const char* name_a, const double a[16], const char* name_b, const double b[16]) {
    double max_d = 0.0;
    int max_i = 0;
    for (int i = 0; i < 16; ++i) {
        const double d = std::fabs(a[i] - b[i]);
        if (d > max_d) {
            max_d = d;
            max_i = i;
        }
    }
    std::cout << "[TRUTH] |" << name_a << " - " << name_b << "| max=" << max_d << " at i=" << max_i
              << "\n";
}
