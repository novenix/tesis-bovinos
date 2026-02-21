#!/bin/bash

# Dirección IP de Cratos
SERVER_IP="10.34.1.43"

echo "🚀 Iniciando script de mantenimiento de conexión para Cratos ($SERVER_IP)..."
echo "Presiona [CTRL+C] para detener."

while true
do
    # Envía 1 solo paquete de ping
    # -c 1: un solo paquete
    # -W 1: timeout de 1 segundo
    ping -c 1 -W 1 $SERVER_IP > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "$(date +'%H:%M:%S') - Ping exitoso: Conexión activa."
    else
        echo "$(date +'%H:%M:%S') - Error de ping: Revisa tu red."
    fi
    
    sleep 10
done
