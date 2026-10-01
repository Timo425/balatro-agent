#!/usr/bin/env bash
# Launch Balatro for the bot bench WITHOUT touching normal play:
#   - Steamodded + BalatroBot load from a separate BotMods folder (Lovely --mod-dir),
#     so Balatro/Mods (balatro-advisor's advisor-live only) stays as it is;
#   - the bot plays on Profile 3; your settings.jkr is backed up and restored on stop,
#     so the next normal launch is Profile 1 with achievements on.
#   tools/bot.sh start | stop | status
set -euo pipefail
B="$HOME/.steam/debian-installation/steamapps/compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro"
BOT="$B/BotMods"
BACKUP="$B/settings.jkr.play-backup"
WIN_BOT='C:\users\steamuser\AppData\Roaming\Balatro\BotMods'
PROFILE=3

running() { pgrep -f -i 'Balatro.exe' >/dev/null; }

set_profile() { # rewrite ["profile"]=N in settings.jkr (raw DEFLATE around a Lua table)
  python3 - "$B/settings.jkr" "$1" <<'EOF'
import re, sys, zlib
path, n = sys.argv[1], sys.argv[2]
t = zlib.decompress(open(path, 'rb').read(), -15).decode()
t2, k = re.subn(r'\["profile"\]=\d+', f'["profile"]={n}', t)
if k != 1:
    sys.exit("settings.jkr: no profile field found")
c = zlib.compressobj(9, zlib.DEFLATED, -15)
open(path, 'wb').write(c.compress(t2.encode()) + c.flush())
EOF
}

case "${1:-}" in
  start)
    running && { echo "Balatro is running: close it first."; exit 1; }
    [ -d "$B/$PROFILE" ] || { echo "No Profile $PROFILE in $B"; exit 1; }
    mkdir -p "$BOT"
    for m in Steamodded Balatrobot; do
      [ -e "$BOT/$m" ] || cp -r "$B/Mods-disabled/$m" "$BOT/"
    done
    [ -e "$BACKUP" ] || cp "$B/settings.jkr" "$BACKUP"
    set_profile "$PROFILE"
    echo "Profile $PROFILE set (your settings backed up). Launching with BotMods…"
    steam -applaunch 2379780 --mod-dir "$WIN_BOT" >/dev/null 2>&1 &
    for _ in $(seq 60); do
      if (exec 3<>/dev/tcp/127.0.0.1/12346) 2>/dev/null; then echo "BalatroBot API is up on 127.0.0.1:12346."; exit 0; fi
      sleep 2
    done
    echo "BalatroBot API didn't come up within 2 minutes (is the game showing Steamodded's mods?)."
    ;;
  stop)
    if running; then pkill -f -i 'Balatro.exe' || true; sleep 3; fi
    if [ -e "$BACKUP" ]; then mv "$BACKUP" "$B/settings.jkr"; echo "Your settings restored (normal play: Profile 1, Mods/ only)."; fi
    ;;
  status)
    echo "BotMods: $(ls "$BOT" 2>/dev/null | tr '\n' ' ')"
    echo "Mods (normal play): $(ls "$B/Mods" | tr '\n' ' ')"
    [ -e "$BACKUP" ] && echo "⚠ Bot session settings are active (Profile $PROFILE): run tools/bot.sh stop before playing."
    running && echo "Balatro is running." || echo "Balatro is not running."
    ;;
  *) echo "usage: tools/bot.sh start|stop|status"; exit 1 ;;
esac
