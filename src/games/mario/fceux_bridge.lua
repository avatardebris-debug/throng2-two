--[[
fceux_bridge.lua -- NES <-> Python RL bridge for FCEUX (file-based IPC)

Communication via two temp files (no sockets needed -- works with stock FCEUX):
  BRIDGE_DIR/obs.txt    Lua WRITES observation + reward + done
  BRIDGE_DIR/act.txt    Lua READS  action from Python

Protocol each frame:
  1. Lua reads RAM → obs, computes reward
  2. Lua writes obs.txt  (atomic: write .tmp then rename)
  3. Lua polls act.txt until Python writes it
  4. Lua reads action, deletes act.txt
  5. Lua presses buttons, advances one frame

Actions:
  0=noop  1=right  2=left  3=jump(A)
  4=run+right(B+R)  5=run+jump+right(B+R+A)
  6=run+left(B+L)   7=run+jump+left(B+L+A)  8=down(D)

Title-screen skip:
  Auto-advances frames until Mario is in actual gameplay (page_x>0 or mario_x>50).

Usage: load this script in FCEUX via File > Lua > New Lua Script Window
       or pass via --lua on command line.
--]]

-- ── CONFIGURATION ───────────────────────────────────────────
local BRIDGE_DIR = os.getenv("FCEUX_BRIDGE_DIR") or "C:/Users/avata/fceux_bridge"
local OBS_FILE   = BRIDGE_DIR .. "/obs.txt"
local ACT_FILE   = BRIDGE_DIR .. "/act.txt"
local READY_FILE = BRIDGE_DIR .. "/ready.txt"
local FRAME_SKIP = 4      -- hold each action this many frames (enables real jumps)
local LOG_EVERY  = 60     -- log every N agent steps (not NES frames)
local _obs_write_ok  = false  -- set true on first successful write

-- SMB (USA) RAM addresses  (verified for standard SMB USA ROM)
local ADDR_MARIO_X    = 0x0086   -- Mario X within current page (screen-relative)
local ADDR_MARIO_Y    = 0x00CE   -- Mario Y position
local ADDR_PAGE_X     = 0x006D   -- horizontal scroll page index
local ADDR_LIVES      = 0x0075   -- lives remaining (2=3 lives, 1=2, 0=1)
local ADDR_SCORE_HI   = 0x07F8   -- score BCD high byte
local ADDR_SCORE_MID  = 0x07F9   -- score BCD mid byte
local ADDR_SCORE_LO   = 0x07FA   -- score BCD low byte
local ADDR_WORLD      = 0x0756   -- world number (0-7)
local ADDR_LEVEL      = 0x0757   -- level number (0-3)
local ADDR_PLAYER_ST  = 0x000E   -- player status: 0x0B=dead/game-over
local ADDR_PLAYER_SZ  = 0x0754   -- player size: 0=small, 1=big/fire

-- ── ACTION → BUTTON TABLE ──────────────────────────────────
local ACTIONS = {
    [0] = {},
    [1] = {right=true},
    [2] = {left=true},
    [3] = {A=true},
    [4] = {B=true, right=true},
    [5] = {B=true, right=true, A=true},
    [6] = {B=true, left=true},
    [7] = {B=true, left=true,  A=true},
    [8] = {down=true},
}

-- ── HELPERS ─────────────────────────────────────────────────

local function bcd_to_int(hi, mid, lo)
    local function bcd(b) return math.floor(b/16)*10 + (b%16) end
    return bcd(hi)*10000 + bcd(mid)*100 + bcd(lo)
end

local function read_obs()
    local mx  = memory.readbyte(ADDR_MARIO_X)
    local my  = memory.readbyte(ADDR_MARIO_Y)
    local px  = memory.readbyte(ADDR_PAGE_X)
    local lv  = memory.readbyte(ADDR_LIVES)
    local hi  = memory.readbyte(ADDR_SCORE_HI)
    local mid = memory.readbyte(ADDR_SCORE_MID)
    local lo  = memory.readbyte(ADDR_SCORE_LO)
    local wo  = memory.readbyte(ADDR_WORLD)
    local le  = memory.readbyte(ADDR_LEVEL)
    local sz  = memory.readbyte(ADDR_PLAYER_SZ)
    local sc  = bcd_to_int(hi, mid, lo)
    return {
        mario_x = mx, mario_y = my, page_x = px,
        lives = lv, score = sc, world = wo, level = le,
        is_big = sz > 0 and 1 or 0,
    }
end

local function is_dead()
    local st = memory.readbyte(ADDR_PLAYER_ST)
    return st == 0x0B or st == 0x06
end

-- ── FILE I/O ────────────────────────────────────────────────

local function file_exists(path)
    local f = io.open(path, "r")
    if f then f:close(); return true end
    return false
end

local function write_obs(obs, reward, done)
    -- Direct write (no rename needed)
    local f = io.open(OBS_FILE, "w")
    if not f then
        if not _obs_write_ok then
            print("[bridge] ERROR: cannot write obs.txt!")
            print("[bridge] Tried: " .. OBS_FILE)
            print("[bridge] Check FCEUX_BRIDGE_DIR env or that the folder exists.")
        end
        return
    end
    _obs_write_ok = true
    f:write(string.format(
        "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f|%.4f|%d\n",
        obs.mario_x / 255.0,
        obs.mario_y / 255.0,
        obs.page_x  / 255.0,
        obs.world   / 7.0,
        obs.level   / 3.0,
        obs.lives   / 3.0,
        obs.score   / 99990.0,
        obs.is_big,
        reward,
        done
    ))
    f:close()
end

local function read_action()
    -- Poll until Python writes act.txt
    local max_polls = 600  -- ~10 seconds at 60fps
    for i = 1, max_polls do
        local f = io.open(ACT_FILE, "r")
        if f then
            local line = f:read("*l")
            f:close()
            os.remove(ACT_FILE)
            if line then
                local n = tonumber(line)
                return n or 0
            end
            return 0
        end
        -- Advance a frame while waiting (keeps FCEUX responsive)
        gui.text(10, 10, "Waiting for Python action...")
        emu.frameadvance()
    end
    print("[bridge] TIMEOUT waiting for action -- returning noop")
    return 0
end

local function wait_for_python()
    -- Wait for Python to create ready.txt (means it's listening)
    print("[bridge] Waiting for Python to start...")
    print("[bridge] Bridge dir: " .. BRIDGE_DIR)
    local frame = 0
    while not file_exists(READY_FILE) do
        frame = frame + 1
        gui.text(10, 10, string.format("Waiting for Python... (%ds)", math.floor(frame/60)))
        if frame % 300 == 0 then
            print(string.format("[bridge] Still waiting for Python... (%ds)", math.floor(frame/60)))
        end
        emu.frameadvance()
    end
    -- Consume the ready file
    os.remove(READY_FILE)
    print("[bridge] Python is ready! Starting bridge.")
    gui.text(10, 10, "Python connected!")
    emu.frameadvance()
end

-- ── TITLE SCREEN SKIP ───────────────────────────────────────

local function skip_title_screen()
    print("[bridge] Fast-forwarding through title screen...")
    gui.text(10, 10, "Skipping title screen...")
    local ff = 0
    while true do
        local px = memory.readbyte(ADDR_PAGE_X)
        local mx = memory.readbyte(ADDR_MARIO_X)
        if px > 0 or mx > 50 then break end
        -- Press Start periodically to get past menus
        if ff % 60 == 30 then
            joypad.set(1, {start=true})
        else
            joypad.set(1, {})
        end
        emu.frameadvance()
        ff = ff + 1
        if ff > 600 then  -- 10 seconds
            print("[bridge] Title skip: proceeding after 10s")
            break
        end
    end
    print(string.format("[bridge] Skipped %d title frames.", ff))
end

-- ── SAVE STATE SLOT ─────────────────────────────────────────
local _savestate = savestate.create()  -- in-memory savestate object

-- ── TITLE SCREEN SKIP (time-based, no RAM guessing) ─────────
-- For the Duck Hunt + Mario combo ROM:
--   Phase 1: press Start a few times to get past Duck Hunt title
--   Phase 2: once on Mario title, press A or Start to start 1P game
--   Phase 3: wait for Mario to actually appear in level

local GAME_MODE   = 0x0756   -- FCEUX game mode register
local TIMER_MODE  = 0x07F3   -- non-zero when in-game timer is running

local function in_gameplay()
    -- Mario's "appear to be alive and moving" check:
    -- Page > 0 means we've scrolled at least once (definitely in level)
    -- lives = 2 usually means first life (0-indexed lives)
    -- score register non-zero can be incidental
    -- Best proxy: timer RAM 07F3 counting (in-game timer only counts in play)
    local timer = memory.readbyte(TIMER_MODE)
    local page  = memory.readbyte(ADDR_PAGE_X)
    local mx    = memory.readbyte(ADDR_MARIO_X)
    local lives = memory.readbyte(ADDR_LIVES)
    -- In-game: lives is 2 (SMB gives 3 lives, stored as 2/1/0)
    -- AND at least mario has an x position
    return (lives <= 2) and (mx > 0 or page > 0)
end

local function skip_to_gameplay()
    print("[bridge] Navigating to gameplay (pressing Start/A to skip menus)...")
    gui.text(10, 10, "Navigating menus...")
    local ff = 0
    local max_ff = 900  -- 15 seconds max
    while ff < max_ff do
        -- Fast multi-press Start sequence to get through title
        local btn = {}
        -- Every 45 frames press Start, at 60 press A as backup
        if ff % 90 < 4 then
            btn = {start=true}
        elseif ff % 90 == 30 then
            btn = {A=true}
        end
        joypad.set(1, btn)
        emu.frameadvance()
        ff = ff + 1

        -- Check if we've landed in gameplay
        if ff > 60 and in_gameplay() then
            -- Extra wait to make sure we're stable
            for i = 1, 30 do
                joypad.set(1, {})
                emu.frameadvance()
            end
            print(string.format("[bridge] Gameplay detected at frame %d!", ff))
            return true
        end
    end
    print("[bridge] WARNING: could not confirm gameplay after 15s, continuing anyway")
    return false
end

-- ── MAIN LOOP ───────────────────────────────────────────────

local function main()
    -- Create bridge directory
    os.execute('mkdir "' .. BRIDGE_DIR .. '" 2>nul')
    -- Clean up stale files
    os.remove(OBS_FILE)
    os.remove(ACT_FILE)
    os.remove(READY_FILE)

    -- Wait for Python
    wait_for_python()

    -- Navigate to gameplay and save state for fast resets
    local have_savestate = false
    skip_to_gameplay()
    print("[bridge] Saving state for fast resets...")
    savestate.save(_savestate)
    have_savestate = true
    print("[bridge] Save state ready.")

    -- State tracking
    local prev_obs    = read_obs()
    local prev_score  = prev_obs.score
    local prev_page   = prev_obs.page_x
    local prev_lives  = prev_obs.lives
    local frame_count = 0
    local frames_since_reset = 0

    -- Send initial observation
    write_obs(prev_obs, 0.0, 0)

    print("[bridge] Main loop started.")

    while true do
        -- Wait for and read Python's action
        local action = read_action()

        -- Check for reset command (action = -1)
        if action == -1 then
            print("[bridge] Reset requested.")
            if have_savestate then
                savestate.load(_savestate)
                -- Let the state settle
                joypad.set(1, {})
                emu.frameadvance()
                emu.frameadvance()
            else
                emu.softreset()
                skip_to_gameplay()
                savestate.save(_savestate)
                have_savestate = true
            end
            prev_obs   = read_obs()
            prev_score = prev_obs.score
            prev_page  = prev_obs.page_x
            prev_lives = prev_obs.lives
            frame_count = 0
            frames_since_reset = 0
            write_obs(prev_obs, 0.0, 0)
        else
            -- Hold action for FRAME_SKIP frames, accumulate reward
            local buttons = ACTIONS[action] or {}
            local accumulated_reward = 0.0
            local final_obs = prev_obs
            local died = false

            for skip = 1, FRAME_SKIP do
                joypad.set(1, buttons)
                emu.frameadvance()
                frame_count = frame_count + 1
                frames_since_reset = frames_since_reset + 1

                local obs = read_obs()
                final_obs = obs

                -- Incremental reward
                local score_delta = (obs.score - prev_score) / 100.0
                local x_progress  = (obs.page_x * 256 + obs.mario_x) -
                                    (prev_page  * 256 + prev_obs.mario_x)
                x_progress = math.max(0, x_progress) * 0.01
                accumulated_reward = accumulated_reward + score_delta + x_progress

                prev_score = obs.score
                prev_page  = obs.page_x
                prev_obs   = obs

                -- Death check (only after grace period)
                if frames_since_reset > 30 then
                    if is_dead() or obs.lives < prev_lives then
                        died = true
                        break
                    end
                end
            end

            local done = died and 1 or 0
            local death_penalty = died and 15.0 or 0.0
            local reward = accumulated_reward - death_penalty

            -- HUD + log
            if (frame_count / FRAME_SKIP) % LOG_EVERY == 0 then
                gui.text(10, 10, string.format("f=%d x=%d pg=%d sc=%d r=%.2f",
                    frame_count, final_obs.mario_x, final_obs.page_x, final_obs.score, reward))
                print(string.format(
                    "[bridge] f=%d  x=%d pg=%d  score=%d  r=%.3f  done=%d  act=%d",
                    frame_count, final_obs.mario_x, final_obs.page_x, final_obs.score,
                    reward, done, action
                ))
            end

            -- Write final observation for Python
            write_obs(final_obs, reward, done)

            prev_lives = final_obs.lives
        end
    end
end

print("[bridge] FCEUX Bridge loaded. Version: file-based IPC + save-state reset")
main()
