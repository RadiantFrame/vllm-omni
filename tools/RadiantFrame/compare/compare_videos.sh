#!/usr/bin/env bash
#
# 双视频对比播放器 - 命令行启动脚本
# 用法: ./compare-videos.sh <视频A路径> <视频B路径>
#
# 示例:
#   ./compare-videos.sh model_a_output.mp4 model_b_output.mp4
#   ./compare-videos.sh /path/to/video1.mp4 /path/to/video2.mp4
#

set -euo pipefail

# ---------- 配置 ----------
# HTML 播放器文件路径（默认与脚本同目录，可自行修改）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="${SCRIPT_DIR}/compare.html"

# 浏览器命令（默认用 xdg-open，可改为 google-chrome / firefox 等）
BROWSER_CMD="${BROWSER_CMD:-xdg-open}"
# --------------------------

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
用法: $(basename "$0") <视频A路径> <视频B路径>

在浏览器中打开双视频对比播放器，并自动加载两个视频。

参数:
  视频A路径    第一个视频文件（模型 A 输出）
  视频B路径    第二个视频文件（模型 B 输出）

选项:
  -h, --help    显示此帮助信息

环境变量:
  BROWSER_CMD   指定浏览器命令（默认: xdg-open）
                示例: BROWSER_CMD=google-chrome ./compare-videos.sh a.mp4 b.mp4

示例:
  $(basename "$0") output_a.mp4 output_b.mp4
  $(basename "$0") /data/videos/model1.mp4 /data/videos/model2.mp4
  BROWSER_CMD=firefox $(basename "$0") a.mp4 b.mp4
EOF
}

# URL 编码函数（Python 可用时用 Python，否则用 sed 处理常见字符）
urlencode() {
    local str="$1"
    if command -v python3 &>/dev/null; then
        python3 -c "import urllib.parse; print(urllib.parse.quote('''$str'''))"
    elif command -v python &>/dev/null; then
        python -c "import urllib; print(urllib.quote('''$str'''))"
    else
        # 回退方案：只处理空格和最常见特殊字符
        echo "$str" | sed 's/ /%20/g; s/#/%23/g; s/?/%3F/g; s/&/%26/g; s/=/%3D/g'
    fi
}

# 检查参数
if [[ $# -lt 2 ]]; then
    echo -e "${RED}错误: 需要两个视频路径参数${NC}" >&2
    echo "" >&2
    usage >&2
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

VIDEO_A="$1"
VIDEO_B="$2"

# 检查 HTML 文件是否存在
if [[ ! -f "$HTML_FILE" ]]; then
    echo -e "${RED}错误: 找不到 HTML 播放器文件: $HTML_FILE${NC}" >&2
    echo "请确认 compare.html 与脚本在同一目录，或修改脚本中的 HTML_FILE 变量。" >&2
    exit 1
fi

# 检查视频文件是否存在
for v in "$VIDEO_A" "$VIDEO_B"; do
    if [[ ! -f "$v" ]]; then
        echo -e "${RED}错误: 视频文件不存在: $v${NC}" >&2
        exit 1
    fi
done

# 转换为绝对路径
VIDEO_A_ABS="$(cd "$(dirname "$VIDEO_A")" && pwd)/$(basename "$VIDEO_A")"
VIDEO_B_ABS="$(cd "$(dirname "$VIDEO_B")" && pwd)/$(basename "$VIDEO_B")"
HTML_ABS="$(cd "$(dirname "$HTML_FILE")" && pwd)/$(basename "$HTML_FILE")"

# ---------- 起本地 HTTP 服务伺服播放器与视频 ----------
# 不走 file://：Chrome 的 file:// 页面默认禁止加载其它本地文件，且
# --allow-file-access-from-files 在已有 Chrome 实例运行时会被忽略
# （新命令只是往现有实例转发标签页）。HTTP 方式对任何浏览器/已开会话都有效。
HTTP_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
HTTP_PID_FILE="/tmp/compare_videos_http.pid"

# 若有上一次残留的服务，先清理
if [[ -f "$HTTP_PID_FILE" ]] && kill -0 "$(cat "$HTTP_PID_FILE")" 2>/dev/null; then
    kill "$(cat "$HTTP_PID_FILE")" 2>/dev/null
fi
# 以文件系统根为伺服目录，这样绝对路径可直接映射成 URL 路径。
# 必须用支持 Range(206) 的服务：python -m http.server 忽略 Range 头，
# Chrome 对无 Range 的源会禁用 seek —— 逐帧/拖动直接失效归零。
python3 "${SCRIPT_DIR}/ranged_server.py" "$HTTP_PORT" / >/dev/null 2>&1 &
HTTP_PID=$!
echo "$HTTP_PID" > "$HTTP_PID_FILE"
sleep 0.5

# URL 编码（http 路径 = 绝对路径去掉开头的 /）
VIDEO_A_ENC="/$(urlencode "${VIDEO_A_ABS#/}")"
VIDEO_B_ENC="/$(urlencode "${VIDEO_B_ABS#/}")"
HTML_ENC="/$(urlencode "${HTML_ABS#/}")"

cleanup_http() {
    kill "$HTTP_PID" 2>/dev/null
    rm -f "$HTTP_PID_FILE"
}
trap cleanup_http INT TERM

# 构造对比播放器 URL（路径均为绝对路径映射到 HTTP 服务根）
URL="http://127.0.0.1:${HTTP_PORT}${HTML_ENC}?a=${VIDEO_A_ENC}&b=${VIDEO_B_ENC}"

echo -e "${GREEN}✓ 视频 A:${NC} $VIDEO_A_ABS"
echo -e "${GREEN}✓ 视频 B:${NC} $VIDEO_B_ABS"
echo -e "${GREEN}✓ 播放器:${NC} $HTML_ABS"
echo ""
echo -e "${YELLOW}正在用 $BROWSER_CMD 打开对比播放器...${NC}"
echo ""

# 用浏览器打开（http URL，无 file:// 限制，任何浏览器/已开会话均可）
if [[ "$BROWSER_CMD" == "xdg-open" ]]; then
    xdg-open "$URL" 2>/dev/null &
else
    "$BROWSER_CMD" "$URL" 2>/dev/null &
fi

# 提示
cat <<EOF
${GREEN}播放器已启动！${NC}

视频通过本地 HTTP 服务加载: http://127.0.0.1:${HTTP_PORT}
（Ctrl+C 退出本脚本会同时停掉该服务；直接关终端则服务残留，
  可用 kill $(cat /tmp/compare_videos_http.pid) 清理）

快捷键:
  空格      播放/暂停
  ← →       快退/快进 5 秒
  , .       上一帧/下一帧
  R         重置到开头
  Tab       切换并排/滑块模式
EOF
