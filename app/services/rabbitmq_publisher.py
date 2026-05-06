"""
RabbitMQ 메시지 발행 서비스
분석 결과를 Spring 서버로 전달
"""
import json
import logging
from datetime import datetime
from typing import Optional

import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from app.core.config import settings

logger = logging.getLogger(__name__)


class RabbitMQPublisher:
    """RabbitMQ 메시지 발행 클래스"""

    def __init__(self):
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel = None

    def connect(self) -> bool:
        """RabbitMQ 서버 연결"""
        try:
            credentials = pika.PlainCredentials(
                settings.rabbitmq_user,
                settings.rabbitmq_password,
            )
            parameters = pika.ConnectionParameters(
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Exchange 선언 (topic 타입)
            self.channel.exchange_declare(
                exchange=settings.rabbitmq_exchange,
                exchange_type="topic",
                durable=True,
            )

            # Queue 선언
            self.channel.queue_declare(
                queue=settings.rabbitmq_queue,
                durable=True,
            )

            # Queue를 Exchange에 바인딩
            self.channel.queue_bind(
                exchange=settings.rabbitmq_exchange,
                queue=settings.rabbitmq_queue,
                routing_key="analysis.*",
            )

            logger.info("RabbitMQ 연결 성공")
            return True

        except AMQPConnectionError as e:
            logger.error(f"RabbitMQ 연결 실패: {e}")
            return False

    def disconnect(self):
        """RabbitMQ 연결 종료"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ 연결 종료")

    def publish_completed(
        self,
        job_id: str,
        user_id: str,
        analysis_result: dict,
        natural_language_response: str,
    ) -> bool:
        """
        분석 완료 메시지 발행

        Args:
            job_id: 작업 ID
            user_id: 사용자 ID
            analysis_result: AI 분석 결과
            natural_language_response: 자연어 응답

        Returns:
            발행 성공 여부
        """
        message = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "completed",
            "result": {
                "analysis": analysis_result,
                "natural_language_response": natural_language_response,
            },
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }

        return self._publish(
            routing_key=settings.rabbitmq_routing_key_completed,
            message=message,
        )

    def publish_failed(
        self,
        job_id: str,
        user_id: str,
        error_code: str,
        error_message: str,
    ) -> bool:
        """
        분석 실패 메시지 발행

        Args:
            job_id: 작업 ID
            user_id: 사용자 ID
            error_code: 에러 코드
            error_message: 에러 메시지

        Returns:
            발행 성공 여부
        """
        message = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "failed",
            "error": {
                "code": error_code,
                "message": error_message,
            },
            "failed_at": datetime.utcnow().isoformat() + "Z",
        }

        return self._publish(
            routing_key=settings.rabbitmq_routing_key_failed,
            message=message,
        )

    def _publish(self, routing_key: str, message: dict) -> bool:
        """실제 메시지 발행 (내부 메서드)"""
        try:
            # 연결 안 되어 있으면 새로 연결
            if not self.connection or self.connection.is_closed:
                if not self.connect():
                    return False

            # 메시지 발행
            self.channel.basic_publish(
                exchange=settings.rabbitmq_exchange,
                routing_key=routing_key,
                body=json.dumps(message, ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 메시지 영속화
                    content_type="application/json",
                ),
            )

            logger.info(
                f"메시지 발행 성공 - routing_key: {routing_key}, job_id: {message.get('job_id')}"
            )
            return True

        except AMQPChannelError as e:
            logger.error(f"메시지 발행 실패: {e}")
            return False

        except Exception as e:
            logger.error(f"메시지 발행 중 알 수 없는 오류: {e}")
            return False

rabbitmq_publisher = RabbitMQPublisher()