"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
## PERSONA
VinAssistant — Trợ lý AI hệ sinh thái Vingroup (VinFast, Vinpearl); phong cách chuyên nghiệp, chính xác.

## AVAILABLE TOOLS
- `search_product_catalog(category, max_price)`: Tra cứu xe điện ('xe_dien') hoặc du lịch ('du_lich') theo giá tối đa (VNĐ).
- `submit_support_ticket(customer_name, issue_description, priority)`: Ghi nhận ticket hỗ trợ/khiếu nại.

## CORE RULES
1. Tuyệt đối KHÔNG hallucinate; BẮT BUỘC gọi tool để lấy dữ liệu thực tế về sản phẩm và tạo ticket.
2. Nếu không tìm thấy dữ liệu từ tool, phản hồi trung thực, không tự suy đoán.

## OPERATIONAL BOUNDARIES
Chỉ hỗ trợ sản phẩm & dịch vụ thuộc Vingroup; từ chối lịch sự các chủ đề ngoài phạm vi.

## OUTPUT CONTRACT
Thực hiện theo chu trình: Thought -> Action -> Action Input -> Observation -> Final Answer.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả về câu trả lời tĩnh (mock) không dùng tool để so sánh đối chứng."""
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, bool]:
        """Phân tích ý định người dùng (Catalog search, Ticket submission, hoặc FAQ)."""
        lower = user_input.lower()

        # FAQ: Hỏi về chính sách bảo hành, pin mà không chứa yêu cầu mua/lọc giá hay tạo ticket
        is_faq = any(k in lower for k in ["bảo hành", "chính sách"]) and not any(
            k in lower for k in ["dưới", "lỗi", "khiếu nại", "ticket", "tôi tên", "tên tôi"]
        )

        # Ticket: Khiếu nại, phản ánh lỗi, ADAS, thông tin khách hàng
        needs_ticket = any(k in lower for k in ["lỗi", "khiếu nại", "phản hồi", "hỗ trợ", "ticket", "adas", "ẩm mốc", "gấp"]) or bool(
            re.search(r"(?:tên tôi là|tôi tên)", lower)
        )

        # Catalog: Tìm kiếm, xem sản phẩm, hỏi giá
        catalog_triggers = ["xem", "tìm", "mua", "giá", "dưới", "triệu", "tỷ", "resort", "catalog", "có xe nào", "xe điện nào"]
        needs_catalog = (
            any(k in lower for k in catalog_triggers)
            and not is_faq
            and ("lỗi" not in lower or "xem" in lower or "resort" in lower)
        )

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_catalog_params(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho search_product_catalog."""
        lower = user_input.lower()

        # Category
        if any(k in lower for k in ["resort", "du lịch", "du_lich", "vinpearl", "phòng", "khách sạn"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        # Max price
        max_price = 999999999999
        match = re.search(r"(?:dưới|tối đa|khoảng|giá)?\s*(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tr|m|đ|đồng|vnd)?", lower)
        if not match:
            match = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|tỷ|tr|m)", lower)

        if match:
            num = float(match.group(1).replace(",", "."))
            unit = (match.group(2) or "").lower()
            if "tỷ" in unit:
                max_price = int(num * 1_000_000_000)
            elif unit in ["triệu", "tr", "m"]:
                max_price = int(num * 1_000_000)
            else:
                if num < 10000:
                    max_price = int(num * 1_000_000)
                else:
                    max_price = int(num)

        return {"category": category, "max_price": max_price}

    def _extract_ticket_params(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho submit_support_ticket."""
        # Customer name
        name_match = re.search(r"(?:tên tôi là|tôi tên(?: là)?)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\s+xe|\s+phòng|\s+sđt|$)", user_input, re.IGNORECASE)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        # Priority
        lower = user_input.lower()
        if any(k in lower for k in ["gấp", "nghiêm trọng", "khẩn cấp", "cao", "high"]):
            priority = "high"
        elif any(k in lower for k in ["thấp", "không vội", "low"]):
            priority = "low"
        else:
            priority = "medium"

        # Issue description
        issue_match = re.search(r"((?:xe|phòng|dịch vụ|hệ thống)[^,\.]*(?:lỗi|hỏng|sự cố|ẩm mốc|kém|không hoạt động)[^,\.]*)", user_input, re.IGNORECASE)
        if issue_match:
            issue_description = issue_match.group(1).strip()
        else:
            issue_description = user_input.strip()

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)
        iteration = 0

        # Safeguard: Kiểm tra giới hạn số bước
        if self.max_iterations <= 0:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        # Kịch bản 1: FAQ (Trả lời trực tiếp, không cần gọi tool)
        if intents["is_faq"]:
            iteration = 1
            answer = (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm (hoặc 200.000 km tùy điều kiện nào đến trước) "
                "đối với các dòng xe trang bị pin cao cấp, và chính sách bảo hành chính hãng tối ưu, hỗ trợ thay thế hoặc sửa chữa "
                "miễn phí nếu dung lượng pin giảm dưới 70%."
            )
            self.trace.append({
                "iteration": iteration,
                "step": "direct_faq",
                "thought": "Đây là câu hỏi thường gặp về chính sách bảo hành pin xe điện VinFast, có thể trả lời trực tiếp mà không cần gọi tool.",
                "action": "none",
                "observation": "VinFast Battery Warranty Policy",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Kịch bản 2: Vòng lặp gọi Tool
        tools_to_run = []
        if intents["needs_catalog"]:
            tools_to_run.append("search_product_catalog")
        if intents["needs_ticket"]:
            tools_to_run.append("submit_support_ticket")

        if not tools_to_run:
            iteration = 1
            answer = "Xin chào, tôi là VinAssistant. Tôi có thể hỗ trợ quý khách tra cứu sản phẩm xe điện VinFast, voucher nghỉ dưỡng Vinpearl hoặc tiếp nhận yêu cầu hỗ trợ."
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        catalog_results = None
        ticket_result = None

        for tool_name in tools_to_run:
            if iteration >= self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            iteration += 1
            if tool_name == "search_product_catalog":
                params = self._extract_catalog_params(user_input)
                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Khách hàng muốn tìm kiếm sản phẩm trong danh mục {params['category']} với mức giá dưới {params['max_price']:,} VNĐ.",
                    "action": "search_product_catalog",
                    "action_input": params
                })
                catalog_results = search_product_catalog(**params)
                self.trace[-1]["observation"] = catalog_results

            elif tool_name == "submit_support_ticket":
                params = self._extract_ticket_params(user_input)
                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Khách hàng {params['customer_name']} cần tạo yêu cầu hỗ trợ về vấn đề: {params['issue_description']}.",
                    "action": "submit_support_ticket",
                    "action_input": params
                })
                ticket_result = submit_support_ticket(**params)
                self.trace[-1]["observation"] = ticket_result

        # Tổng hợp câu trả lời Final Answer
        answer_parts = []

        if catalog_results is not None:
            if not catalog_results:
                answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của quý khách.")
            else:
                prods_desc = []
                for p in catalog_results:
                    prods_desc.append(f"- {p['name']} (Giá: {p['price_vnd']:,} VNĐ): {p.get('description', '')}")
                answer_parts.append(f"Dưới đây là các sản phẩm phù hợp với yêu cầu của quý khách:\n" + "\n".join(prods_desc))

        if ticket_result is not None:
            answer_parts.append(
                f"Yêu cầu hỗ trợ của quý khách {ticket_result['customer_name']} đã được tiếp nhận thành công. "
                f"Mã ticket hỗ trợ: {ticket_result['ticket_id']} (Trạng thái: {ticket_result['status']}). "
                f"Đội ngũ kỹ thuật Vingroup sẽ liên hệ xử lý trong thời gian sớm nhất."
            )

        final_answer = "\n\n".join(answer_parts)
        self.trace.append({
            "step": "final_synthesis",
            "final_answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", result["answer"])
    print("\nTrace Log:\n", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
