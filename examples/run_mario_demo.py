"""
run_mario_demo.py -- Train numpy agent on Mario ASCII + render gameplay.

Pure numpy, no PyTorch. Trains on curriculum levels (procedural + GAN mix).
Outputs ASCII gameplay to terminal AND saves an HTML visualization.

Usage:
    python examples/run_mario_demo.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.games.mario.mario_simulator import MarioSimulator, Action
from src.games.mario.mario_adapter import MarioAdapter
from src.games.mario.mario_curriculum import MarioCurriculum
from src.games.mario.mario_agent import MarioRLAgent

EPISODES = 100
MAX_STEPS = 300
LOG_INTERVAL = 10
RENDER_EPISODES = [0, 10, 50, 99]  # Episodes to capture for replay

def main():
    print("=" * 60)
    print("  MARIO ASCII -- NUMPY RL TRAINING + VISUAL DEMO")
    print("=" * 60)

    adapter = MarioAdapter()
    curriculum = MarioCurriculum(
        start_tier=1,
        advance_threshold=0.7,
        window_size=20,
        seed=42,
    )

    agent = MarioRLAgent(
        obs_dim=adapter.obs_dim,  # 378
        n_actions=adapter.n_actions,  # 6
        hidden1=128,
        hidden2=64,
        lr=3e-4,
        gamma=0.99,
        rollout_length=128,
    )

    print(f"  Obs dim: {adapter.obs_dim}")
    print(f"  Actions: {adapter.n_actions}")
    print(f"  Network: 378 -> 128(tanh) -> 64(tanh) -> 6 actions + value")
    print()

    # Collect replay frames for visualization
    replay_data = {}
    episode_rewards = []
    episode_wins = []
    t_start = time.perf_counter()

    for ep in range(EPISODES):
        level = curriculum.next_level()
        obs = adapter.reset(level)
        agent.reset()

        ep_reward = 0.0
        frames = []
        capture = ep in RENDER_EPISODES

        for step in range(MAX_STEPS):
            # Capture frame
            if capture:
                frame_text = adapter.render(viewport=False)
                stats = adapter.stats()
                frames.append({
                    "ascii": frame_text,
                    "step": step,
                    "mario_pos": stats["mario_pos"],
                    "coins": stats["coins"],
                    "score": round(stats["score"], 2),
                    "alive": stats["alive"],
                })

            action = agent.step(obs)
            obs, reward, done, info = adapter.step(action)
            ep_reward += reward
            agent.learn(reward, done)

            if done:
                if capture:
                    frame_text = adapter.render(viewport=False)
                    stats = adapter.stats()
                    frames.append({
                        "ascii": frame_text,
                        "step": step + 1,
                        "mario_pos": stats["mario_pos"],
                        "coins": stats["coins"],
                        "score": round(stats["score"], 2),
                        "alive": stats["alive"],
                        "won": level.won,
                    })
                break

        won = level.won
        progress = level.max_x_reached / max(1, level.width)
        episode_rewards.append(ep_reward)
        episode_wins.append(int(won))

        curriculum.record_result(won=won, progress=progress, steps=step + 1, level=level)

        if capture:
            replay_data[ep] = {
                "frames": frames,
                "won": won,
                "reward": round(ep_reward, 2),
                "progress": round(progress, 2),
                "tier": curriculum.tier,
                "width": level.width,
            }

        if curriculum.should_advance():
            old_tier = curriculum.tier
            new_tier = curriculum.advance()
            print(f"  >>> ADVANCED tier {old_tier} -> {new_tier}")

        if ep % LOG_INTERVAL == 0 or ep == EPISODES - 1:
            recent_r = episode_rewards[-LOG_INTERVAL:]
            recent_w = episode_wins[-LOG_INTERVAL:]
            elapsed = time.perf_counter() - t_start
            print(f"  Ep {ep:3d} | tier={curriculum.tier} "
                  f"| avg_r={np.mean(recent_r):+6.2f} "
                  f"| win={np.mean(recent_w):.0%} "
                  f"| progress={progress:.2f} "
                  f"| {elapsed:.0f}s")

    # ── Generate HTML visualization ─────────────────────────
    elapsed = time.perf_counter() - t_start
    print()
    print(f"  Training: {EPISODES} episodes in {elapsed:.1f}s")
    print(f"  Final win rate: {np.mean(episode_wins[-20:]):.0%}")
    print()

    html_path = os.path.join(os.path.dirname(__file__), "mario_replay.html")
    generate_html(replay_data, html_path, episode_rewards, episode_wins)
    print(f"  Replay saved to: {html_path}")
    print("  Open in browser to watch the agent play!")
    print("=" * 60)


def generate_html(replay_data, path, rewards, wins):
    """Generate an HTML file with ASCII game replay animation."""

    # Build replay JSON
    import json
    replay_json = json.dumps(replay_data, indent=2)

    # Rewards chart data
    chart_data = json.dumps({
        "rewards": [round(r, 2) for r in rewards],
        "wins": wins,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mario ASCII RL -- Game Replay</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a2e;
    color: #e0e0ff;
    font-family: 'Courier New', monospace;
    min-height: 100vh;
    padding: 20px;
  }}
  h1 {{
    text-align: center;
    color: #ff6b6b;
    font-size: 28px;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(255,107,107,0.3);
  }}
  .subtitle {{
    text-align: center;
    color: #7878ff;
    margin-bottom: 20px;
    font-size: 14px;
  }}
  .controls {{
    text-align: center;
    margin: 15px 0;
  }}
  .controls button {{
    background: #2a2a5e;
    color: #e0e0ff;
    border: 1px solid #4a4a8e;
    padding: 8px 20px;
    margin: 0 5px;
    cursor: pointer;
    font-family: monospace;
    font-size: 14px;
    border-radius: 4px;
    transition: all 0.2s;
  }}
  .controls button:hover {{
    background: #3a3a7e;
    border-color: #6a6aae;
  }}
  .controls button.active {{
    background: #ff6b6b;
    border-color: #ff9b9b;
    color: white;
  }}
  .controls select {{
    background: #2a2a5e;
    color: #e0e0ff;
    border: 1px solid #4a4a8e;
    padding: 8px 15px;
    font-family: monospace;
    font-size: 14px;
    border-radius: 4px;
  }}
  #game-container {{
    max-width: 800px;
    margin: 0 auto;
    background: #1a1a3e;
    border: 2px solid #3a3a6e;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }}
  #game-screen {{
    font-size: 16px;
    line-height: 1.3;
    white-space: pre;
    color: #a0ffa0;
    min-height: 300px;
    padding: 10px 0;
  }}
  #game-screen .ground {{ color: #8B4513; }}
  #game-screen .mario {{ color: #ff0000; font-weight: bold; }}
  #game-screen .enemy {{ color: #ff6600; }}
  #game-screen .coin {{ color: #ffff00; }}
  #game-screen .flag {{ color: #00ff00; font-weight: bold; }}
  #game-screen .pipe {{ color: #228B22; }}
  #game-screen .brick {{ color: #cd853f; }}
  #game-screen .question {{ color: #ffd700; font-weight: bold; }}
  #game-screen .platform {{ color: #808080; }}
  #game-screen .pit {{ color: #4a0000; }}
  #status-bar {{
    color: #7878ff;
    padding: 8px 0;
    font-size: 13px;
    border-top: 1px solid #3a3a6e;
    margin-top: 10px;
  }}
  #progress-bar {{
    width: 100%;
    height: 6px;
    background: #2a2a5e;
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
  }}
  #progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #ff6b6b, #ffaa00, #00ff00);
    transition: width 0.1s;
    border-radius: 3px;
  }}
  #chart-container {{
    max-width: 800px;
    margin: 20px auto;
    background: #1a1a3e;
    border: 2px solid #3a3a6e;
    border-radius: 8px;
    padding: 20px;
  }}
  canvas {{ width: 100%; height: 200px; }}
  .result-banner {{
    text-align: center;
    font-size: 24px;
    padding: 10px;
    margin-top: 10px;
    border-radius: 4px;
    display: none;
  }}
  .result-banner.win {{
    background: rgba(0,255,0,0.15);
    color: #00ff00;
    border: 1px solid #00ff00;
    display: block;
  }}
  .result-banner.lose {{
    background: rgba(255,0,0,0.15);
    color: #ff4444;
    border: 1px solid #ff4444;
    display: block;
  }}
</style>
</head>
<body>
<h1>MARIO ASCII RL</h1>
<div class="subtitle">ThrongletCell-compatible numpy agent learning to play</div>

<div class="controls">
  <label>Episode:
    <select id="ep-select" onchange="loadEpisode(this.value)"></select>
  </label>
  <button id="btn-play" onclick="togglePlay()">PLAY</button>
  <button onclick="stepFrame(-1)">PREV</button>
  <button onclick="stepFrame(1)">NEXT</button>
  <label>Speed:
    <select id="speed-select" onchange="setSpeed(this.value)">
      <option value="500">Slow</option>
      <option value="200" selected>Normal</option>
      <option value="80">Fast</option>
      <option value="30">Turbo</option>
    </select>
  </label>
</div>

<div id="game-container">
  <div id="game-screen">Loading...</div>
  <div id="status-bar"></div>
  <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
  <div id="result-banner" class="result-banner"></div>
</div>

<div id="chart-container">
  <h3 style="color:#ff6b6b; margin-bottom:10px;">Training Progress</h3>
  <canvas id="reward-chart"></canvas>
</div>

<script>
const REPLAYS = {replay_json};
const CHART = {chart_data};

let currentEp = null;
let currentFrame = 0;
let playing = false;
let playTimer = null;
let speed = 200;

// Color map: character -> CSS class
const CHAR_COLORS = {{
  'M': 'mario', 'G': 'enemy', 'T': 'enemy', 'P': 'enemy',
  'L': 'enemy', 's': 'enemy', 'E': 'enemy',
  'o': 'coin', 'F': 'flag', '#': 'ground',
  '[': 'pipe', ']': 'pipe', 'B': 'brick',
  '?': 'question', '=': 'platform', '_': 'pit',
}};

// Color-code ASCII per character (avoids regex cascading corruption)
function colorize(ascii) {{
  // Split into grid lines and status line
  const lines = ascii.split('\\n');
  const statusLine = lines.length > 0 && lines[lines.length-1].trim().startsWith('Mario')
    ? lines.pop() : null;

  let html = '';
  for (const line of lines) {{
    for (const ch of line) {{
      const cls = CHAR_COLORS[ch];
      if (cls) {{
        html += '<span class="' + cls + '">' + ch + '</span>';
      }} else {{
        html += (ch === '<' ? '&lt;' : ch === '>' ? '&gt;' : ch === '&' ? '&amp;' : ch);
      }}
    }}
    html += '\\n';
  }}
  // Status line as plain text (no colorization)
  if (statusLine) {{
    html += '<span style="color:#7878ff;font-size:12px">' +
            statusLine.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</span>';
  }}
  return html;
}}

function init() {{
  const sel = document.getElementById('ep-select');
  for (const ep of Object.keys(REPLAYS).sort((a,b) => +a - +b)) {{
    const d = REPLAYS[ep];
    const opt = document.createElement('option');
    opt.value = ep;
    opt.textContent = `Ep ${{ep}} (${{d.won ? 'WIN' : 'LOSE'}}, tier=${{d.tier}}, r=${{d.reward}})`;
    sel.appendChild(opt);
  }}
  const firstEp = Object.keys(REPLAYS)[0];
  if (firstEp) loadEpisode(firstEp);
  drawChart();
}}

function loadEpisode(ep) {{
  currentEp = ep;
  currentFrame = 0;
  stopPlay();
  showFrame();
}}

function showFrame() {{
  if (!currentEp || !REPLAYS[currentEp]) return;
  const data = REPLAYS[currentEp];
  const frames = data.frames;
  if (currentFrame >= frames.length) currentFrame = frames.length - 1;
  if (currentFrame < 0) currentFrame = 0;

  const f = frames[currentFrame];
  document.getElementById('game-screen').innerHTML = colorize(f.ascii);
  document.getElementById('status-bar').textContent =
    `Step: ${{f.step}} | Pos: (${{f.mario_pos[0]}},${{f.mario_pos[1]}}) | ` +
    `Coins: ${{f.coins}} | Score: ${{f.score}} | ` +
    `Frame: ${{currentFrame+1}}/${{frames.length}} | ` +
    `Episode: ${{currentEp}} (tier ${{data.tier}}, ${{data.width}}-wide)`;

  const pct = ((currentFrame + 1) / frames.length * 100).toFixed(0);
  document.getElementById('progress-fill').style.width = pct + '%';

  const banner = document.getElementById('result-banner');
  if (currentFrame === frames.length - 1) {{
    if (data.won) {{
      banner.className = 'result-banner win';
      banner.textContent = 'WIN! Reward: ' + data.reward;
    }} else {{
      banner.className = 'result-banner lose';
      banner.textContent = 'DIED. Progress: ' + (data.progress * 100).toFixed(0) + '%';
    }}
  }} else {{
    banner.className = 'result-banner';
  }}
}}

function stepFrame(delta) {{
  if (!currentEp) return;
  currentFrame += delta;
  showFrame();
}}

function togglePlay() {{
  if (playing) {{ stopPlay(); }} else {{ startPlay(); }}
}}

function startPlay() {{
  playing = true;
  document.getElementById('btn-play').textContent = 'PAUSE';
  document.getElementById('btn-play').classList.add('active');
  playTimer = setInterval(() => {{
    if (!currentEp || !REPLAYS[currentEp]) return;
    if (currentFrame < REPLAYS[currentEp].frames.length - 1) {{
      currentFrame++;
      showFrame();
    }} else {{
      stopPlay();
    }}
  }}, speed);
}}

function stopPlay() {{
  playing = false;
  document.getElementById('btn-play').textContent = 'PLAY';
  document.getElementById('btn-play').classList.remove('active');
  if (playTimer) clearInterval(playTimer);
  playTimer = null;
}}

function setSpeed(ms) {{ speed = parseInt(ms); if (playing) {{ stopPlay(); startPlay(); }} }}

function drawChart() {{
  const canvas = document.getElementById('reward-chart');
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth;
  canvas.height = 200;
  const w = canvas.width, h = canvas.height;

  ctx.fillStyle = '#1a1a3e';
  ctx.fillRect(0, 0, w, h);

  if (!CHART.rewards || CHART.rewards.length === 0) return;

  const rewards = CHART.rewards;
  const wins = CHART.wins;
  const n = rewards.length;
  const maxR = Math.max(...rewards) || 1;
  const minR = Math.min(...rewards);

  // Draw reward line
  ctx.beginPath();
  ctx.strokeStyle = '#ff6b6b';
  ctx.lineWidth = 1.5;
  for (let i = 0; i < n; i++) {{
    const x = (i / n) * w;
    const y = h - ((rewards[i] - minR) / (maxR - minR + 0.01) * (h - 20)) - 10;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  // Draw win dots
  for (let i = 0; i < n; i++) {{
    if (wins[i]) {{
      const x = (i / n) * w;
      ctx.fillStyle = '#00ff00';
      ctx.beginPath();
      ctx.arc(x, 10, 3, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}

  // Labels
  ctx.fillStyle = '#7878ff';
  ctx.font = '11px monospace';
  ctx.fillText('Reward (red line) | Wins (green dots)', 10, h - 5);
}}

window.onload = init;
window.onresize = drawChart;
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
