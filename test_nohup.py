import time
import datetime
import sys

print(f"🚀 Script de prueba iniciado a las {datetime.datetime.now()}")
sys.stdout.flush()

for i in range(10):
    print(f"[{i+1}/10] HOLA MUNDO - El servidor me deja vivir en segundo plano. Time: {datetime.datetime.now()}")
    sys.stdout.flush()
    time.sleep(2)

print("🏁 Prueba finalizada con éxito.")
