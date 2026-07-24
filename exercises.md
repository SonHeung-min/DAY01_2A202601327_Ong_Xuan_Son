# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay các dòng giữ chỗ bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature = 0.0, phản hồi thường ổn định, an toàn và ít thay đổi giữa các lần gọi. Khi tăng lên 0.5 và 1.0, câu trả lời đa dạng hơn về cách diễn đạt và ví dụ; đến 1.5 thì nội dung có xu hướng sáng tạo hơn nhưng cũng dễ lan man hoặc kém nhất quán hơn. Quy luật chung là temperature càng cao thì độ ngẫu nhiên càng lớn, còn temperature thấp phù hợp khi cần câu trả lời chắc và dễ dự đoán.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, em sẽ đặt temperature khoảng 0.2-0.4. Mục tiêu của chatbot này là trả lời nhất quán, đúng chính sách và ít bịa thông tin, nên không cần quá sáng tạo; vẫn để một chút temperature giúp câu trả lời tự nhiên hơn thay vì quá máy móc.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload có 10.000 x 3 x 350 = 10.500.000 token đầu ra mỗi ngày. Theo bảng giá trong `template.py`, GPT-4o output là 0.010 USD/1K token còn GPT-4o-mini là 0.0006 USD/1K token, nên GPT-4o đắt hơn khoảng 16,7 lần cho phần output. GPT-4o xứng đáng khi cần suy luận phức tạp, chất lượng cao hoặc câu trả lời ảnh hưởng lớn đến người dùng; GPT-4o-mini phù hợp cho hỏi đáp phổ thông, phân loại đơn giản, tóm tắt ngắn hoặc tác vụ số lượng lớn cần tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, phản hồi thường ngắn hơn, dùng từ đơn giản và hay ví blockchain như một cuốn sổ ghi chép chung mà nhiều người cùng xem. Với persona chuyên gia tài chính, câu trả lời dài hơn, dùng các thuật ngữ như sổ cái phân tán, cơ chế đồng thuận, tính bất biến, tài sản số hoặc giao dịch phi tập trung. System prompt làm thay đổi vai trò, giọng văn, độ sâu kỹ thuật và kiểu ví dụ mà model ưu tiên. Vì vậy cùng một câu hỏi, model có thể điều chỉnh cách trả lời cho đúng đối tượng người đọc.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Em dùng đoạn văn tiếng Việt 96 từ; `count_tokens` với model `gpt-4o` cho kết quả 116 token, còn ước lượng theo `số từ / 0.75` là 128 token. Hai con số chênh khoảng 9,4%, trong lần đo này tiktoken thấp hơn công thức ước lượng thô. Tiếng Việt thường tốn nhiều token vì có dấu, nhiều âm tiết tách bằng khoảng trắng và một số từ/cụm từ không khớp gọn với các mảnh token đã học như tiếng Anh, nên tokenizer có thể phải chia nhỏ hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài hoặc người dùng cần cảm giác hệ thống đang trả lời ngay, ví dụ chatbot tư vấn, trợ lý viết nội dung, giải thích bài học hoặc sinh mã nhiều đoạn. Nó giảm thời gian chờ cảm nhận được vì người dùng đọc được phần đầu trong lúc phần sau vẫn đang sinh. Non-streaming phù hợp hơn khi cần xử lý kết quả trọn gói trước khi hiển thị, ví dụ phân loại, trích xuất JSON, kiểm thử tự động, ghi log, hoặc các tác vụ ngắn mà độ trễ không đáng kể.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp client giảm tốc độ retry dần dần khi API đang quá tải, nhờ đó server có thời gian phục hồi và tránh bị dồn thêm yêu cầu lỗi. So với delay cố định 1 giây, backoff thích nghi tốt hơn với lỗi kéo dài vì mỗi lần thử sau cách xa hơn. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng có thể tạo thành từng đợt request đồng loạt, làm hệ thống tiếp tục nghẽn và khiến lỗi 429/timeout kéo dài hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona là trợ giảng AI thân thiện cho sinh viên mới học API. System prompt: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn, rõ ràng bằng tiếng Việt; ưu tiên ví dụ thực tế, chỉ nêu thuật ngữ kỹ thuật khi thật cần và luôn nhắc người học kiểm chứng kết quả quan trọng." Cụm "trả lời ngắn gọn, rõ ràng" giúp CLI không sinh quá dài, tiết kiệm token và dễ đọc trong terminal. Yêu cầu "bằng tiếng Việt" giữ trải nghiệm nhất quán với người học trong lab.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại là assistant chỉ giữ 3 lượt hội thoại gần nhất, chưa có bộ nhớ dài hạn và chưa có cơ chế kiểm duyệt hoặc kiểm chứng nguồn. Một cải thiện cụ thể là lưu tóm tắt cuộc trò chuyện sau mỗi vài lượt vào một biến `summary`, rồi gửi summary này cùng system prompt ở các lượt sau để giữ ngữ cảnh dài hơn mà không làm history phình quá lớn. Khi triển khai, có thể thêm hàm `summarize_history(history)` gọi model rẻ hơn hoặc dùng rule đơn giản, sau đó cắt history như hiện tại nhưng vẫn giữ phần tóm tắt trong messages.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
