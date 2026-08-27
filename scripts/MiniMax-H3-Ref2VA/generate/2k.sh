#!/bin/bash

# 创建视频任务
curl --request POST \
  --url https://api.minimaxi.com/v2/video_generation \
  --header "Authorization: Bearer ${MINIMAX_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data '
{
  "model": "MiniMax-H3",
  "content": [
    {
      "type": "text",
      "text": "史诗级太空歌剧院线预告：女舰长独自站在巨大观景窗前，最后一支舰队正在集结并跃迁离去，强光爆闪、舰桥震动，她被留在原地。"
    }
  ],
  "resolution": "2K",
  "duration": 5,
  "ratio": "16:9"
}
'
# 查询任务: 按 task_id 查询最近 7 天内单个视频生成、H3-Context-IR 或视频再生成任务的状态与结
curl --request GET \
  --url https://api.minimaxi.com/v2/query/video_generation/427387636240809 \
  --header "Authorization: Bearer ${MINIMAX_API_KEY}"