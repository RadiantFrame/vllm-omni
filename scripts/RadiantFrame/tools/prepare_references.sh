#!/usr/bin/env bash
# =============================================================================
# prepare_references.sh
#
# LE SSERAFIM Ref2VA benchmark —— 参考资源下载与准备脚本
# 面向 MiniMax H3 Ref2VA 测试用例：
#   Image  1..5 : 五位成员 identity reference（Sakura / Kim Chaewon /
#                 Huh Yunjin / Kazuha / Hong Eunchae）
#   Image  6    : 五人团体 group reference（队形 / 造型 / 视觉概念）
#   Video  1..3 : 编舞 / 舞台表演 / 运镜 三个视频参考
#   Audio  1..3 : 人声 / 音乐 / 现场氛围 三个音频参考
#
# 对齐 MiniMax H3 Ref2VA 输入约束：
#   Images ≤ 9（本包 6）
#   Videos ≤ 3 clips，每个 2–15s，同类合计 ≤ 15s
#   Audio  ≤ 3 clips，每个 2–15s，同类合计 ≤ 15s
#   全部输入文件总数 ≤ 12（本包 6 图 + 3 视频 + 3 音频 = 12）
#
# 目录结构：
#   raw/      完整素材（保留，可复用于重新截取）
#   videos/   提交给 H3 的短视频片段（默认每段 5s，3 段共 15s）
#   audio/    提交给 H3 的短音频片段（默认每段 5s，3 段共 15s，与视频同步）
#   images/   6 张图片
#   manifest/ references_manifest.json
#
# 用法:
#   ./prepare_references.sh [--out DIR] [--images-only] [--videos-only]
#                           [--dry-run] [--force] [--max-h 2160] [--audio FMT]
#                           [--clip-secs 5] [--start-offset S]
#
# 依赖: python3, ffmpeg, ffprobe, curl（或 wget）
#
# 视频/音频下载方案（已实测可行）:
#   统一走 B 站纯 API 下载器 download_bili_api.py（view + WBI 签名 playurl +
#   直链下载 + ffmpeg 合并）。该方案不抓取 B 站视频网页，因此对云服务器 /
#   数据中心 IP 被 B 站 HTTP 412 风控的环境完全免疫（已实测通过）。
#   视频参考与 live 现场均为内置、经 API 核验的官方 bvid，无需运行时搜索。
# =============================================================================
set -uo pipefail

# --------------------------- 1. 配置区（按需修改） --------------------------
# 输出根目录（默认: 脚本同目录下 references/）
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/references}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BILI_API_PY="$SCRIPT_DIR/download_bili_api.py"  # B 站纯 API 下载器（需与脚本同目录）

# 下载行为
FORCE=0            # 1 = 强制重新下载（默认跳过已存在且校验通过的文件）
RETRIES=2          # 单个文件失败后的重试次数
DRY_RUN=0
MODE_IMAGES=1
MODE_VIDEOS=1

# ---- H3 片段裁剪参数 ----
CLIP_SECS=5        # 每个参考片段时长（秒）。3 个视频 + 3 个音频合计均须 ≤ 15s，
                   # 故 CLIP_SECS 最大 5；超出自动收缩并提示
START_OFFSET=-1    # 统一起始偏移（秒）；-1 = 自动取每段视频/音频的中间段
                   # （避开片头 logo 与片尾 credits，取代表性段落）
# 每个视频独立的起始偏移（秒），-1 = 沿用 START_OFFSET 行为（统一值或自动中段）。
# 例：--start-offsets -1,42,10 → V1 自动中段、V2 从 42s、V3 从 10s 截取
# （对应音频 Audio1/Audio2 与视频同窗口，保持音画同步）
V_STARTS=(-1 -1 -1)
KEEP_VIDEO_AUDIO=0  # 1 = 保留参考视频内嵌音轨（会占用 H3 音频总时长 15s 预算，慎用）

# ---- 6 张图片参考（按 Ref2VA Image1..6 顺序）----
# 支持两种来源：远程 URL(http/https) 或 本地绝对路径(以 / 开头，直接复制)。
# 默认指向已人工核验身份的官方/高清素材。建议在同一概念期(如 ANTIFRAGILE/HOT)
# 内自选成套，以保证 5 人 identity 与团体风格的视觉一致性，可随时替换。
IMG_SOURCES=(
  "https://aka.doubaocdn.com/s/Tdj7aiSzz8"   # Image1 Sakura
  "https://aka.doubaocdn.com/s/zo8Hf8X5qw"   # Image2 Kim Chaewon
  "https://aka.doubaocdn.com/s/LW2UCRbZ8r"   # Image3 Huh Yunjin
  "https://aka.doubaocdn.com/s/8iSTBDUwJu"   # Image4 Kazuha
  "https://aka.doubaocdn.com/s/6IMxswVXzA"   # Image5 Hong Eunchae
  "https://aka.doubaocdn.com/s/woWTJQLGjt"   # Image6 五人团体 group concept
)
IMG_NAMES=(sakura kim_chaewon huh_yunjin kazuha hong_eunchae group)
IMG_FILES=()  # 记录实际保存的文件名，供 manifest 使用

# ---- 3 个视频参考：B 站直链（均经 B 站 API 核验的官方 MV / 编舞 / 舞台）----
BILI_V1_URL="https://www.bilibili.com/video/BV1mP411A7Qo"  # 编舞参考: ANTIFRAGILE Choreography ver.
BILI_V2_URL="https://www.bilibili.com/video/BV1dPPXzNEKk"  # 表演参考: UNFORGIVEN 官方MV(4K修复)
BILI_V3_URL="https://www.bilibili.com/video/BV1tx4y1k7Bu"  # 运镜参考: EASY 官方MV
V1_NAME="v1_antifragile_choreo"
V2_NAME="v2_unforgiven_perform"
V3_NAME="v3_easy_cinema"

# ---- 3 个音频参考 ----
# Audio1(人声)   <- 取自 V1 视频音轨（ANTIFRAGILE 官方音轨，人声清晰）
# Audio2(音乐)   <- 取自 V2 视频音轨（UNFORGIVEN，K-pop 制作/编曲）
# Audio3(现场氛围) <- 取自 LIVE 视频音轨（Coachella 完整现场，含观众/舞台环境声）
#   用 API 的 audio-only 模式直接拉音轨（无需下载整段视频）
BILI_LIVE_URL="https://www.bilibili.com/video/BV1Pr421574f"  # 240414 科切拉 1080p 完整舞台

# ---- 输出参数 ----
# 注意：MiniMax H3 Ref2VA 上传层只接受 .wav 或 .mp3 音频（m4a/flac 会被 400 拒绝），
# 故默认 mp3；wav 亦可用。请勿选 m4a/flac 除非先转码。
AUDIO_FMT="${AUDIO_FMT:-mp3}"   # mp3(推荐,H3 兼容) / wav(H3 兼容) / m4a|flac(需自行转码)
VIDEO_MAXH="${VIDEO_MAXH:-2160}" # 视频最高高度: 1080 / 2160 / 4320

# ============================ 工具函数 ======================================
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[警告]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[错误]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }
have_cmd() { command -v "$1" >/dev/null 2>&1; }

check_deps() {
  local miss=()
  have_cmd python3 || miss+=(python3)
  have_cmd ffmpeg || miss+=(ffmpeg)
  have_cmd ffprobe || miss+=(ffprobe)
  if ! have_cmd curl && ! have_cmd wget; then miss+=(curl或wget); fi
  if [[ ${#miss[@]} -gt 0 ]]; then
    die "缺少依赖: ${miss[*]}。请先安装，例如: apt-get install python3 ffmpeg curl"
  fi
}

mkdirs() {
  mkdir -p "$OUT_DIR"/{raw,images,videos,audio,manifest} || die "无法创建输出目录 $OUT_DIR"
}

# 简易下载（curl 优先，失败退回 wget）
dl() {
  local url="$1" dest="$2"
  if have_cmd curl; then
    curl -fsSL --retry 3 --connect-timeout 20 -o "$dest" "$url"
  else
    wget -q -O "$dest" "$url"
  fi
}

# 校验是否为有效图片（非空 + 图像类型 + 尺寸）
valid_image() {
  local f="$1"
  [[ -s "$f" ]] || return 1
  local ct
  ct=$(file -b --mime-type "$f" 2>/dev/null || true)
  case "$ct" in
    image/*) ;;
    *) return 1 ;;
  esac
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
    -of csv=s=x:p=0 "$f" >/dev/null 2>&1 || return 1
  return 0
}

# B 站纯 API 下载（view + WBI 签名 playurl + 直链下载 + ffmpeg 合并）
# mode=video: 下载音视频流并合并为 mp4；mode=audio: 仅下载音频为 m4a
bili_api_download() {
  local url="$1" dest="$2" mode="$3"
  [[ -s "$BILI_API_PY" ]] || { warn "  缺少 $BILI_API_PY（需与脚本同目录），跳过下载"; return 1; }
  log "  [B站API] $mode 模式下载: $url"
  if python3 "$BILI_API_PY" --url "$url" --out "$dest" \
      --maxh "$VIDEO_MAXH" --mode "$mode" --tmpdir "${TMPDIR:-/tmp}"; then
    return 0
  fi
  return 1
}

# 下载单个完整视频到 raw/（video 模式，含重试）
download_raw_video() {
  local name="$1" bili_url="$2" dest="$3"
  if [[ -f "$dest" ]] && [[ "$FORCE" -eq 0 ]]; then
    log "跳过 $name（完整版已存在: $dest）"
    return 0
  fi
  local ok=0
  for _ in $(seq 1 $((RETRIES+1))); do
    if bili_api_download "$bili_url" "$dest" "video"; then
      ok=1
      break
    fi
    warn "  $name 下载失败，重试"
    sleep 2
  done
  return $((1-ok))
}

# 计算裁剪起始点：--start-offset 指定则用之；否则取素材中段 (dur - len)/2
# 返回: 起始秒数（保留 1 位小数）
calc_start() {
  local dur="$1" len="$2"
  if [[ "$START_OFFSET" -ge 0 ]]; then
    echo "$START_OFFSET"
  else
    awk -v d="$dur" -v l="$len" 'BEGIN{ s=(d-l)/2; if(s<0)s=0; printf "%.1f", s }'
  fi
}

# 从完整素材裁剪出 H3 兼容的短视频片段（h264 + yuv420p + aac，最兼容）
clip_video() {
  local src="$1" dst="$2" label="$3" start="${4:--1}"
  [[ -s "$src" ]] || { warn "跳过视频片段 $label：源缺失 $src"; return 1; }
  [[ -f "$dst" ]] && [[ "$FORCE" -eq 0 ]] && { log "跳过视频片段 $label（已存在）"; return 0; }
  local dur len
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src" | cut -d. -f1)
  len="$CLIP_SECS"
  if (( dur <= len )); then
    warn "  $label 源仅 ${dur}s，不足 ${len}s，改用完整长度"
    len="$dur"
  fi
  # 调用方传入的起始点（<0 = 自动取素材中段）
  if (( $(awk -v s="$start" 'BEGIN{print (s<0)?1:0}') )); then
    start=$(calc_start "$dur" "$len")
  fi
  log "  裁剪视频片段 $label <- $src (${start}s +${len}s)"
  # H3 的 Ref2VA 会把参考视频内嵌音轨与独立音频合并计入同一 15s 音频预算
  # （pipeline: embedded + external audio_lengths ≤ 600 ticks）。本基准的音频
  # 角色全部由独立 mp3 承担，故默认剥离视频音轨；--keep-video-audio 可保留。
  local audio_args=(-an)
  [[ "$KEEP_VIDEO_AUDIO" -eq 1 ]] && audio_args=(-c:a aac -b:a 320k)
  ffmpeg -hide_banner -loglevel error -y -ss "$start" -t "$len" -i "$src" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    "${audio_args[@]}" -movflags +faststart "$dst" || return 1
  return 0
}

# 从完整素材裁剪出 H3 兼容的短音频片段；与对应视频使用同一时间窗口保持同步
clip_audio() {
  local src="$1" dst="$2" label="$3" start="$4" len="$5"
  [[ -s "$src" ]] || { warn "跳过音频片段 $label：源缺失 $src"; return 1; }
  [[ -f "$dst" ]] && [[ "$FORCE" -eq 0 ]] && { log "跳过音频片段 $label（已存在）"; return 0; }
  log "  裁剪音频片段 $label <- $src (${start}s +${len}s)"
  local codec_args=()
  case "$AUDIO_FMT" in
    wav)  codec_args=(-c:a pcm_s16le) ;;
    flac) codec_args=(-c:a flac) ;;
    mp3)  codec_args=(-c:a libmp3lame -b:a 320k) ;;
    *)    codec_args=(-c:a aac -b:a 320k) ;;
  esac
  ffmpeg -hide_banner -loglevel error -y -ss "$start" -t "$len" -i "$src" -vn \
    "${codec_args[@]}" "$dst" || return 1
  return 0
}

# 校验视频时长（秒，取整）
video_duration() {
  ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null \
    | cut -d. -f1
}

# ============================ 主流程 =======================================
main() {
  check_deps
  mkdirs
  log "输出目录: $OUT_DIR"

  # 片段时长合法性：每个 clip 2–15s，3 段合计 ≤ 15s
  if [[ "$CLIP_SECS" -lt 2 ]]; then
    warn "CLIP_SECS=$CLIP_SECS 小于 H3 下限 2s，强制设为 2"
    CLIP_SECS=2
  elif [[ "$CLIP_SECS" -gt 5 ]]; then
    warn "CLIP_SECS=$CLIP_SECS：3 段视频/音频合计需 ≤ 15s，强制收缩为 5"
    CLIP_SECS=5
  fi
  log "片段参数: 每段 ${CLIP_SECS}s（3 段合计 $((CLIP_SECS*3))s ≤ 15s）"
  echo

  # ---------- A. 图片参考 (Image1..6) ----------
  if [[ "$MODE_IMAGES" -eq 1 ]]; then
    log "阶段 A: 下载 6 张图片参考"
    for i in "${!IMG_SOURCES[@]}"; do
      src="${IMG_SOURCES[$i]}"
      name="${IMG_NAMES[$i]}"
      n=$((i+1))
      ext="jpg"
      dest="$OUT_DIR/images/img${n}_${name}.${ext}"
      existing=$(compgen -G "$OUT_DIR/images/img${n}_${name}.*" 2>/dev/null | head -1)
      if [[ -n "$existing" ]] && valid_image "$existing" && [[ "$FORCE" -eq 0 ]]; then
        log "  跳过 Image${n} ${name}（已存在且有效: $(basename "$existing")）"
        IMG_FILES[$i]="images/$(basename "$existing")"
        continue
      fi
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  [dry] Image${n} ${name} <- $src"
        IMG_FILES[$i]="images/img${n}_${name}.jpg"
        continue
      fi
      tmp="$dest.tmp"
      rm -f "$tmp"
      log "  Image${n} ${name} <- $src"
      ok=0
      for _ in $(seq 1 $((RETRIES+1))); do
        if dl "$src" "$tmp"; then
          mt=$(file -b --mime-type "$tmp" 2>/dev/null)
          case "$mt" in image/png) ext="png";; image/webp) ext="webp";; esac
          dest="$OUT_DIR/images/img${n}_${name}.${ext}"
          mv -f "$tmp" "$dest"
          if valid_image "$dest"; then
            dim=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$dest")
            log "    完成 ($(du -h "$dest" | cut -f1), ${dim})"
            ok=1
            IMG_FILES[$i]="images/$(basename "$dest")"
            break
          else
            rm -f "$dest"; warn "  Image${n} 内容非有效图片，重试"
          fi
        else
          warn "  Image${n} 下载失败，重试"
        fi
        sleep 1
      done
      [[ "$ok" -eq 1 ]] || err "  Image${n} ${name} 下载失败"
    done
    echo
  fi

  # ---------- B. 完整视频参考 (下载到 raw/) ----------
  if [[ "$MODE_VIDEOS" -eq 1 ]]; then
    log "阶段 B: 下载 3 个完整视频 + 1 个 live 音轨到 raw/"
    declare -a VBILI=( "$BILI_V1_URL" "$BILI_V2_URL" "$BILI_V3_URL")
    declare -a VNM=(   "$V1_NAME"     "$V2_NAME"     "$V3_NAME")
    for j in 0 1 2; do
      name="${VNM[$j]}"
      dest="$OUT_DIR/raw/${name}_full.mp4"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "  [dry] 完整视频 ${name} <- ${VBILI[$j]}"
        continue
      fi
      # 迁移旧布局：若 videos/ 下存在旧完整版（>15s），移到 raw/ 避免重复下载
      old="$OUT_DIR/videos/${name}.mp4"
      if [[ ! -s "$dest" ]] && [[ -s "$old" ]]; then
        old_dur=$(video_duration "$old")
        if [[ -n "$old_dur" ]] && (( old_dur > 15 )); then
          log "  迁移旧完整版 $old -> $dest"
          mv -f "$old" "$dest"
        fi
      fi
      if download_raw_video "$name" "${VBILI[$j]}" "$dest"; then
        dur=$(video_duration "$dest")
        log "  完整视频 $name 完成（$(du -h "$dest" | cut -f1), ${dur}s）"
      else
        err "  完整视频 $name 下载失败"
      fi
    done
    # live 完整音轨（audio 模式，供 Audio3）
    live_full="$OUT_DIR/raw/live_full.m4a"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "  [dry] live 完整音轨 <- $BILI_LIVE_URL"
    elif [[ ! -s "$live_full" ]] || [[ "$FORCE" -eq 1 ]]; then
      if bili_api_download "$BILI_LIVE_URL" "$live_full" "audio"; then
        log "  live 完整音轨完成（$(du -h "$live_full" | cut -f1)）"
      else
        err "  live 完整音轨下载失败"
      fi
    else
      log "  live 完整音轨已存在，跳过"
    fi
    echo

    # ---------- C. 裁剪 H3 兼容片段（videos/ + audio/，同步截取）----------
    log "阶段 C: 裁剪 H3 兼容短片段（每段 ${CLIP_SECS}s）"
    declare -a RNM=( "$V1_NAME" "$V2_NAME" "$V3_NAME" )
    # 各段起始点（与对应音频同步）
    declare -a VSTART=()
    for j in 0 1 2; do
      full="$OUT_DIR/raw/${RNM[$j]}_full.mp4"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        VSTART[$j]=0.0
        continue
      fi
      if [[ -s "$full" ]]; then
        dur=$(video_duration "$full")
        # 独立偏移 (--start-offsets) > 统一偏移 (--start-offset) > 自动中段
        if (( V_STARTS[$j] >= 0 )); then
          VSTART[$j]="${V_STARTS[$j]}"
          if awk -v s="${VSTART[$j]}" -v l="$CLIP_SECS" -v d="$dur" 'BEGIN{exit !(s+l>d)}'; then
            warn "  ${RNM[$j]} 起始 ${VSTART[$j]}s + ${CLIP_SECS}s 超出源时长 ${dur}s，将被截短"
          fi
        else
          VSTART[$j]=$(calc_start "$dur" "$CLIP_SECS")
        fi
      else
        VSTART[$j]=0.0
        warn "  缺少完整视频 ${RNM[$j]}，无法裁剪"
      fi
    done
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "  [dry] videos/audio 片段：v1/v2/v3 与 audio1/audio2 同步从 ${VSTART[0]}/${VSTART[1]}/${VSTART[2]}s 截 ${CLIP_SECS}s"
      log "  [dry] audio3 从 live 中段截 ${CLIP_SECS}s"
    else
      # 视频片段（起始点来自上方 VSTART：独立偏移 > 统一偏移 > 自动中段）
      clip_video "$OUT_DIR/raw/${V1_NAME}_full.mp4" "$OUT_DIR/videos/${V1_NAME}.mp4" "Video1" "${VSTART[0]}"
      clip_video "$OUT_DIR/raw/${V2_NAME}_full.mp4" "$OUT_DIR/videos/${V2_NAME}.mp4" "Video2" "${VSTART[1]}"
      clip_video "$OUT_DIR/raw/${V3_NAME}_full.mp4" "$OUT_DIR/videos/${V3_NAME}.mp4" "Video3" "${VSTART[2]}"
      # 音频片段（Audio1/2 与对应视频同窗口，保持音画同步）
      clip_audio "$OUT_DIR/raw/${V1_NAME}_full.mp4" "$OUT_DIR/audio/audio1_vocal.${AUDIO_FMT}" \
        "Audio1 人声" "${VSTART[0]}" "$CLIP_SECS"
      clip_audio "$OUT_DIR/raw/${V2_NAME}_full.mp4" "$OUT_DIR/audio/audio2_music.${AUDIO_FMT}" \
        "Audio2 音乐" "${VSTART[1]}" "$CLIP_SECS"
      # Audio3 现场氛围：从 live 完整音轨中段截取
      if [[ -s "$OUT_DIR/raw/live_full.m4a" ]]; then
        live_dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT_DIR/raw/live_full.m4a" | cut -d. -f1)
        live_start=$(calc_start "$live_dur" "$CLIP_SECS")
        clip_audio "$OUT_DIR/raw/live_full.m4a" "$OUT_DIR/audio/audio3_live_ambience.${AUDIO_FMT}" \
          "Audio3 现场氛围" "$live_start" "$CLIP_SECS"
      else
        warn "  缺少 live 完整音轨，Audio3 跳过"
      fi
    fi
    echo
  fi

  # ---------- D. 汇总清单 ----------
  log "阶段 D: 生成清单 manifest"
  manifest="$OUT_DIR/manifest/references_manifest.json"
  {
    echo '{'
    echo '  "benchmark": "LE SSERAFIM Ref2VA MV generation benchmark",'
    echo '  "model_under_test": "MiniMax H3 Ref2VA",'
    echo '  "h3_input_constraints": {'
    echo '    "images": "<=9", "videos": "<=3 clips, each 2-15s, total <=15s",'
    echo '    "audio": "<=3 clips, each 2-15s, total <=15s", "total_files": "<=12"'
    echo '  },'
    echo "  \"clip_seconds\": ${CLIP_SECS},"
    echo '  "references": {'
    echo '    "images": {'
    for i in "${!IMG_NAMES[@]}"; do
      n=$((i+1))
      f="${IMG_FILES[$i]:-images/img${n}_${IMG_NAMES[$i]}.jpg}"
      echo "      \"Image${n}\": \"$f\","
    done
    echo '      "note": "Image1-5=member identity; Image6=group concept"'
    echo '    },'
    echo '    "videos": {'
    echo "      \"Video1\": \"videos/${V1_NAME}.mp4\",   // ${CLIP_SECS}s, 编舞 (B站: ${BILI_V1_URL})"
    echo "      \"Video2\": \"videos/${V2_NAME}.mp4\",   // ${CLIP_SECS}s, 表演 (B站: ${BILI_V2_URL})"
    echo "      \"Video3\": \"videos/${V3_NAME}.mp4\"    // ${CLIP_SECS}s, 运镜 (B站: ${BILI_V3_URL})"
    echo '    },'
    echo '    "audio": {'
    echo "      \"Audio1\": \"audio/audio1_vocal.${AUDIO_FMT}\",   // ${CLIP_SECS}s, <- V1 同窗口"
    echo "      \"Audio2\": \"audio/audio2_music.${AUDIO_FMT}\",   // ${CLIP_SECS}s, <- V2 同窗口"
    echo "      \"Audio3\": \"audio/audio3_live_ambience.${AUDIO_FMT}\"  // ${CLIP_SECS}s, <- Coachella 现场(B站: ${BILI_LIVE_URL})"
    echo '    }'
    echo '  },'
    echo '  "generation_prompt": "Generate a new 12-second cinematic K-pop girl-group music video featuring the five members of LE SSERAFIM: Sakura, Kim Chaewon, Huh Yunjin, Kazuha, and Hong Eunchae. IMPORTANT: Create a new original MV sequence. Do not reproduce or copy any existing LE SSERAFIM music video shot-for-shot."'
    echo '}'
  } > "$manifest"
  log "manifest -> $manifest"

  # ---------- 汇总 ----------
  echo
  log "========== 汇总 =========="
  find "$OUT_DIR" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.mp4' -o -iname '*.m4a' -o -iname '*.wav' -o -iname '*.flac' \) 2>/dev/null | sort | while read -r f; do
    rel=${f#"$OUT_DIR"/}
    printf '  %-46s %8s\n' "$rel" "$(du -h "$f" | cut -f1)"
  done
  echo
  log "完成。参考资源按 Ref2VA 约定就绪于: $OUT_DIR"
  log "注意：提交给 H3 的是 videos/ 与 audio/ 下的短片段（各 ${CLIP_SECS}s）；raw/ 为完整素材供复现/重截。"
  log "下一步：将 images/videos/audio 三组参考按顺序传给 MiniMax H3 Ref2VA 接口（Image1..6 / Video1..3 / Audio1..3）。"
}

# ---------------------------- 参数解析 --------------------------------------
usage() {
  cat <<EOF
用法: $0 [选项]

  --out DIR        输出目录（默认: 脚本同目录/references）
  --images-only    只处理 6 张图片
  --videos-only    只处理 3 视频 + 3 音频（下载+裁剪）
  --dry-run        只打印计划，不实际下载
  --force          覆盖已存在文件
  --max-h N        视频最高高度，如 1080/2160（默认 2160）
  --audio FMT      音频格式 m4a/wav/flac（默认 m4a）
  --clip-secs N    每个参考片段秒数，2-5（默认 5；3 段合计 ≤ 15s）
  --start-offset S 统一起始偏移秒（默认自动取素材中段）
  --start-offsets S1,S2,S3
                   每个视频独立起始秒，-1=该段沿用默认；如 -1,42,10
  --keep-video-audio
                   保留参考视频内嵌音轨（默认剥离；内嵌音轨占用 H3 音频 15s 总预算）
  -h, --help       显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT_DIR="$2"; shift 2 ;;
    --images-only) MODE_VIDEOS=0; shift ;;
    --videos-only) MODE_IMAGES=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --max-h) VIDEO_MAXH="$2"; shift 2 ;;
    --audio) AUDIO_FMT="$2"; shift 2 ;;
    --clip-secs) CLIP_SECS="$2"; shift 2 ;;
    --start-offset) START_OFFSET="$2"; shift 2 ;;
    --keep-video-audio) KEEP_VIDEO_AUDIO=1; shift ;;
    --start-offsets)
      IFS=',' read -r -a V_STARTS <<< "$2"
      [[ ${#V_STARTS[@]} -eq 3 ]] || { err "--start-offsets 需要恰好 3 个值（S1,S2,S3）"; usage; exit 1; }
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) err "未知参数: $1"; usage; exit 1 ;;
  esac
done

main
