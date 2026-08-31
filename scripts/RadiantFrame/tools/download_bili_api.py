#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_bili_api.py —— B 站纯 API 下载器（绕过网页/yt-dlp 的 HTTP 412 风控）

原理：不抓取视频网页，只调用 B 站公开 API：
  1) 访问 bilibili.com 获取 buvid3/buvid4 cookie
  2) nav API 取 wbi 密钥，对 playurl 请求做 WBI 签名
  3) view API 取 cid，playurl API 取音/视频流直链（dash）
  4) curl 下载音/视频流，ffmpeg 合并（video 模式）或仅存音频（audio 模式）

适用于数据中心 IP 被 B 站风控（412）但 API 可用的场景（已实测通过）。

用法:
  python3 download_bili_api.py --url BV1mP411A7Qo --out out.mp4 [--maxh 1080] [--mode video|audio]
  依赖: ffmpeg（PATH 中可用）
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HOME = "https://www.bilibili.com"

# wbi mixin key 置换表（B 站官方公开常量）
_MIXIN_TAB = [46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,33,9,
              42,19,29,28,14,39,12,38,41,13,37,48,7,16,24,55,40,61,26,17,0,1,
              60,51,30,4,22,25,54,21,56,59,6,63,57,62,11,36,20,34,44,52]


def log(msg):
    sys.stderr.write(msg + "\n")


def fetch(url, cookies, referer=HOME, timeout=60, headers=None):
    h = {"User-Agent": UA, "Referer": referer}
    if cookies:
        h["Cookie"] = "; ".join("%s=%s" % (k, v) for k, v in cookies.items())
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url, cookies, referer=HOME):
    return json.loads(fetch(url, cookies, referer).decode("utf-8"))


def get_cookies():
    """访问 B 站首页，取 buvid3/buvid4 等 cookie（API 风控需要）"""
    jar = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(jar)
    req = urllib.request.Request(HOME, headers={"User-Agent": UA})
    opener.open(req, timeout=30)
    return {c.name: c.value for c in jar.cookiejar}


def get_wbi_mixin(cookies):
    """从 nav API 取 wbi 密钥并生成 mixin key"""
    nav = get_json("https://api.bilibili.com/x/web-interface/nav", cookies)
    d = nav.get("data") or {}
    img = (d.get("wbi_img") or {}).get("img_url", "")
    sub = (d.get("wbi_img") or {}).get("sub_url", "")
    img_key = img.rsplit("/", 1)[-1].split(".")[0]
    sub_key = sub.rsplit("/", 1)[-1].split(".")[0]
    if not img_key or not sub_key:
        raise RuntimeError("nav API 未返回 wbi 密钥（可能需要更新风控处理）")
    s = img_key + sub_key
    return "".join(s[i] for i in _MIXIN_TAB)[:32]


def wbi_sign(params, mixin):
    params["wts"] = int(time.time())
    q = urllib.parse.urlencode(sorted(params.items()))
    params["w_rid"] = hashlib.md5((q + mixin).encode("utf-8")).hexdigest()
    return params


def resolve_bvid(text):
    text = text.strip()
    m = re.search(r"(BV[0-9A-Za-z]{10})", text)
    if m:
        return m.group(1)
    if re.fullmatch(r"av\d+", text):
        return text  # 由 av 处理（view API 也支持 av）
    raise ValueError("无法从输入解析出 bvid: %s" % text)


def get_playinfo(bvid, cookies, mixin, maxh=1080, audio_only=False):
    if bvid.startswith("BV"):
        view = get_json("https://api.bilibili.com/x/web-interface/view?bvid=%s" % bvid, cookies)
    else:
        view = get_json("https://api.bilibili.com/x/web-interface/view?aid=%s" % bvid[2:], cookies)
    if view.get("code") != 0:
        raise RuntimeError("view API 失败: %s" % view.get("message"))
    data = view["data"]
    cid = data["cid"]
    params = wbi_sign({"bvid": bvid, "cid": cid, "qn": 80, "fnval": 16, "fourk": 1}, mixin)
    url = "https://api.bilibili.com/x/player/playurl?" + urllib.parse.urlencode(params)
    pl = get_json(url, cookies)
    if pl.get("code") != 0:
        raise RuntimeError("playurl API 失败: %s" % pl.get("message"))
    dash = (pl.get("data") or {}).get("dash") or {}

    audio = None
    if dash.get("audio"):
        audio = max(dash["audio"], key=lambda s: s.get("bandwidth", 0))

    video = None
    if not audio_only:
        # 优先 h264(avc)，其次 hevc，最后其它；取 <=maxh 中带宽最高的
        cands = [s for s in dash.get("video", []) if s.get("height", 9999) <= maxh]
        if not cands:
            cands = dash.get("video", [])
        for pref in ("avc1", "hev1", "hvc1", ""):
            grp = [s for s in cands if pref == "" or (s.get("codecs") or "").startswith(pref)]
            if grp:
                video = max(grp, key=lambda s: s.get("bandwidth", 0))
                break
    if not audio_only and video is None:
        raise RuntimeError("未找到可用视频流")
    if audio is None:
        raise RuntimeError("未找到可用音频流")
    return {"title": data.get("title", ""), "cid": cid,
            "video": video, "audio": audio}


def download(url, dest, referer=HOME, retries=3):
    for i in range(retries):
        try:
            data = fetch(url, None, referer=referer, timeout=120)
            with open(dest, "wb") as f:
                f.write(data)
            if os.path.getsize(dest) > 0:
                return dest
        except Exception as e:
            log("  下载失败(%s) 第%d次: %s" % (os.path.basename(dest), i + 1, e))
            time.sleep(2)
    raise RuntimeError("下载失败: %s" % dest)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description="bilibili API downloader")
    ap.add_argument("--url", required=True, help="bilibili 视频页 URL 或 bvid")
    ap.add_argument("--out", required=True, help="输出文件路径")
    ap.add_argument("--maxh", type=int, default=1080, help="视频最大高度（默认 1080）")
    ap.add_argument("--mode", choices=["video", "audio"], default="video")
    ap.add_argument("--tmpdir", default="/tmp", help="临时目录")
    args = ap.parse_args()

    bvid = resolve_bvid(args.url)
    log("解析 bvid: %s（mode=%s, maxh=%d）" % (bvid, args.mode, args.maxh))
    cookies = get_cookies()
    mixin = get_wbi_mixin(cookies)
    info = get_playinfo(bvid, cookies, mixin, args.maxh, args.mode == "audio")
    log("命中: %s (cid=%s)" % (info["title"][:60], info["cid"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    os.makedirs(args.tmpdir, exist_ok=True)
    if args.mode == "audio":
        au = info["audio"]
        tmp = os.path.join(args.tmpdir, "bili_%s_audio.m4s" % bvid)
        log("下载音频流 (%s) ..." % au.get("id"))
        download(au["baseUrl"], tmp)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", tmp, "-c", "copy", args.out])
        log("完成: %s" % args.out)
        return

    vu = info["video"]
    au = info["audio"]
    tmpv = os.path.join(args.tmpdir, "bili_%s_v.m4s" % bvid)
    tmpa = os.path.join(args.tmpdir, "bili_%s_a.m4s" % bvid)
    log("下载视频流 (id=%s, %sx%s, codec=%s) ..."
        % (vu.get("id"), vu.get("width"), vu.get("height"), vu.get("codecs")))
    download(vu["baseUrl"], tmpv)
    log("下载音频流 (id=%s) ..." % au.get("id"))
    download(au["baseUrl"], tmpa)
    log("ffmpeg 合并 -> %s" % args.out)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", tmpv, "-i", tmpa, "-c", "copy", args.out])
    for t in (tmpv, tmpa):
        if os.path.exists(t):
            os.remove(t)
    log("完成: %s" % args.out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        sys.stderr.write("错误: %s\n" % e)
        sys.exit(1)
