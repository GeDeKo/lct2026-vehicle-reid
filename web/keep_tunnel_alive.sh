#!/usr/bin/env bash
# Держит проброс порта 8800 (API на A100) живым весь показ жюри.
# Если туннель падает — сам переподнимает за секунды. Запускать один раз
# и оставить работать в фоне (или в отдельном окне терминала).
#
# Пароль НЕ хранится в этом файле (и не должен попасть в git) — передайте
# его через переменную окружения:
#   JUMP_PASS='...' ./keep_tunnel_alive.sh
set -u

PORT=8800
REMOTE_HOST=172.16.2.93
JUMP_USER=ansible-server
JUMP_HOST=89.125.151.113
JUMP_PORT=40007

if [ -z "${JUMP_PASS:-}" ]; then
    echo "Ошибка: задайте пароль через переменную окружения перед запуском:" >&2
    echo "  JUMP_PASS='...' $0" >&2
    exit 1
fi

echo "$(date '+%H:%M:%S') keep_tunnel_alive: старт, слежу за localhost:$PORT"

while true; do
    if ! nc -z -w2 localhost "$PORT" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') туннель мёртв — переподнимаю..."
        /usr/bin/expect -c "
            set timeout 15
            spawn ssh -p $JUMP_PORT -fN -L $PORT:$REMOTE_HOST:$PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes $JUMP_USER@$JUMP_HOST
            expect {
                \"password:\" { send \"$JUMP_PASS\r\"; exp_continue }
                eof
            }
        " > /dev/null 2>&1
        sleep 2
        if nc -z -w2 localhost "$PORT" 2>/dev/null; then
            echo "$(date '+%H:%M:%S') туннель поднят"
        else
            echo "$(date '+%H:%M:%S') не удалось поднять, повторю через 5с"
        fi
    fi
    sleep 5
done
