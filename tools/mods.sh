#!/usr/bin/env bash
# Switch Balatro's Mods folder between the bot bench and normal play.
#   bot  = Steamodded + BalatroBot loaded (needed by this repo; turns achievements OFF)
#   play = only Lovely mods such as balatro-advisor's advisor-live (achievements stay on)
# Both use the same Steam launch option: WINEDLLOVERRIDES="version=n,b" %command%
# (Lovely is the loader; Steamodded is itself a Lovely mod.) Restart the game after switching.
set -euo pipefail
B="$HOME/.steam/debian-installation/steamapps/compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro"
mkdir -p "$B/Mods-disabled"
move() { [ -e "$1/$3" ] && mv "$1/$3" "$2/"; true; }
case "${1:-}" in
  bot)  for m in Steamodded Balatrobot; do move "$B/Mods-disabled" "$B/Mods" "$m"; done ;;
  play) for m in Steamodded Balatrobot; do move "$B/Mods" "$B/Mods-disabled" "$m"; done ;;
  *) echo "usage: tools/mods.sh bot|play"; echo "Mods now: $(ls "$B/Mods" | tr '\n' ' ')"; exit 1 ;;
esac
echo "Mods now: $(ls "$B/Mods" | tr '\n' ' ')"
