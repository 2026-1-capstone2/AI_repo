"""
ngrok 터널 실행 스크립트
"""
from pyngrok import ngrok
import time

# 8000번 포트 터널 열기
public_url = ngrok.connect(8000)

print("=" * 60)
print(f"🌐 공개 URL: {public_url.public_url}")
print(f"📚 Swagger:  {public_url.public_url}/docs")
print(f"❤️  Health:   {public_url.public_url}/health")
print("=" * 60)
print("\n팀원에게 공유할 URL:")
print(f"   {public_url.public_url}/docs\n")
print("종료하려면 Ctrl+C 누르세요...")

# 종료까지 대기
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n터널 종료 중...")
    ngrok.kill()
    print("종료 완료!")