"""
Omni-Channel Support Agent (OCS)

Responsibilities:
- Detect language (en/ar)
- Classify intent
- Generate the primary customer-facing response
- Manage channel normalization
"""

from __future__ import annotations

from pydantic import BaseModel
import structlog

from agents.base_agent import BaseAgent
from agents.omni_channel_support.intent_classifier import classify_intent
from agents.omni_channel_support.language_detector import detect_language
from api.schemas import OCSInput, OCSOutput, SupportedLanguage

logger = structlog.get_logger()


# ── Response Templates ────────────────────────────────────────────────────────

_RESPONSE_TEMPLATES: dict[str, dict[str, str]] = {
    "billing_inquiry": {
        "en": "I understand you have a billing question. Let me look into your account details and provide clarification.",
        "ar": "أفهم أن لديك سؤال حول الفوترة. دعني أراجع تفاصيل حسابك وأقدم لك التوضيح.",
    },
    "technical_support": {
        "en": "I'm sorry you're experiencing technical difficulties. Let me help you troubleshoot this issue.",
        "ar": "أنا آسف لأنك تواجه صعوبات تقنية. دعني أساعدك في حل هذه المشكلة.",
    },
    "account_management": {
        "en": "I can help you with your account settings. Let me guide you through the process.",
        "ar": "يمكنني مساعدتك في إعدادات حسابك. دعني أرشدك خلال العملية.",
    },
    "complaint": {
        "en": "I sincerely apologize for the inconvenience. Your concern is very important to us, and I'll do my best to resolve it.",
        "ar": "أعتذر بشدة عن الإزعاج. قلقك مهم جداً بالنسبة لنا، وسأبذل قصارى جهدي لحله.",
    },
    "escalation_request": {
        "en": "I understand you'd like to speak with a human agent. Let me connect you right away.",
        "ar": "أفهم أنك ترغب في التحدث مع وكيل بشري. دعني أوصلك على الفور.",
    },
    "order_status": {
        "en": "Let me check the status of your order for you right away.",
        "ar": "دعني أتحقق من حالة طلبك فوراً.",
    },
    "cancellation": {
        "en": "I understand you'd like to cancel. Before proceeding, may I ask if there's anything we can do to improve your experience?",
        "ar": "أفهم أنك ترغب في الإلغاء. قبل المتابعة، هل يمكنني أن أسأل إذا كان هناك شيء يمكننا فعله لتحسين تجربتك؟",
    },
    "refund_request": {
        "en": "I understand you'd like a refund. Let me review your order and process this for you.",
        "ar": "أفهم أنك ترغب في استرداد المبلغ. دعني أراجع طلبك وأعالج هذا الأمر لك.",
    },
    "greeting": {
        "en": "Hello! Welcome to our customer support. How can I assist you today?",
        "ar": "مرحباً! أهلاً بك في خدمة العملاء. كيف يمكنني مساعدتك اليوم؟",
    },
    "farewell": {
        "en": "Thank you for contacting us! If you need anything else, don't hesitate to reach out. Have a great day!",
        "ar": "شكراً لتواصلك معنا! إذا احتجت أي شيء آخر، لا تتردد في التواصل. أتمنى لك يوماً سعيداً!",
    },
    "general_inquiry": {
        "en": "Thank you for reaching out. I'll do my best to help answer your question.",
        "ar": "شكراً لتواصلك. سأبذل قصارى جهدي للمساعدة في الإجابة على سؤالك.",
    },
    "product_information": {
        "en": "I'd be happy to provide information about our products. Let me find the details for you.",
        "ar": "يسعدني تقديم معلومات حول منتجاتنا. دعني أجد التفاصيل لك.",
    },
    "feedback": {
        "en": "Thank you for sharing your feedback with us. We truly value your input and will use it to improve our services.",
        "ar": "شكراً لمشاركتك ملاحظاتك معنا. نحن نقدر حقاً مساهمتك وسنستخدمها لتحسين خدماتنا.",
    },
    "unknown": {
        "en": "Thank you for your message. Could you please provide more details so I can assist you better?",
        "ar": "شكراً لرسالتك. هل يمكنك تقديم المزيد من التفاصيل حتى أتمكن من مساعدتك بشكل أفضل؟",
    },
}


class OCSAgent(BaseAgent):
    """Omni-Channel Support Agent."""

    def __init__(self) -> None:
        super().__init__(agent_name="OCS")

    async def process(self, input_data: BaseModel) -> OCSOutput:
        """
        1. Detect language
        2. Classify intent
        3. Generate response
        """
        data: OCSInput = input_data  # type: ignore[assignment]

        # Step 1 — Language detection
        detected_lang = detect_language(data.customer_message)
        language = SupportedLanguage(detected_lang)

        # Step 2 — Intent classification
        intent, confidence = await classify_intent(data.customer_message)

        # Step 3 — Response generation
        templates = _RESPONSE_TEMPLATES.get(intent, _RESPONSE_TEMPLATES["unknown"])
        response_text = templates.get(detected_lang, templates.get("en", ""))

        # Step 4 — Check for explicit escalation
        escalation_flag = intent == "escalation_request"

        self.logger.info(
            "ocs_processed",
            interaction_id=data.interaction_id,
            intent=intent,
            confidence=round(confidence, 3),
            language=detected_lang,
            escalation_flag=escalation_flag,
        )

        return OCSOutput(
            response_text=response_text,
            intent=intent,
            suggested_faq_ids=[],
            escalation_flag=escalation_flag,
            language=language,
        )
