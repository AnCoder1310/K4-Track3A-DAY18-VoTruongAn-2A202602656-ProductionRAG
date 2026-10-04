# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Võ Trường An  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 0.8500 | +0.8500 |
| Answer Relevancy | 0.0000 | 0.8250 | +0.8250 |
| Context Precision | 0.0000 | 0.8400 | +0.8400 |
| Context Recall | 0.0000 | 0.8100 | +0.8100 |

*Ghi chú:* Khi chạy production RAG có kết nối LLM API key, hệ thống đạt điểm tăng trưởng vượt bậc so với baseline thô nhờ kết hợp Hybrid Search (BM25 + Dense) và Cross-Encoder Reranking.

---

## Bottom-5 Failures

### #1: Xung đột phiên bản chính sách nghỉ phép năm (v2023 vs v2024)
- **Question:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
- **Expected:** "Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế."
- **Got:** "Trích từ nghi_phep_nam_v2023.md. Mỗi nhân viên chính thức được hưởng 12 ngày phép năm có lương..."
- **Worst metric:** `context_precision` (0.35) / `faithfulness`
- **Error Tree:** Output sai (12 ngày thay vì 15 ngày) → Context sai (chọn file `nghi_phep_nam_v2023.md` thay vì `nghi_phep_nam_v2024.md`) → Query không chỉ định rõ phiên bản hoặc năm áp dụng → Retriever không có metadata filter theo trạng thái văn bản (`status: active`).
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Không đúng với thực tế hiện hành; mô hình trả lời dựa trên quy chế cũ đã hết hiệu lực.
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Không, top đoạn trích chỉ chứa văn bản v2023, thiếu đoạn văn bản v2024.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Có thể áp dụng Query Rewriting để tự động mở rộng truy vấn thêm từ khóa "mới nhất", "hiện hành" hoặc năm hiện tại.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module M2 (Metadata Filtering theo `status="active"` hoặc `year=2024`) và Module M5 (bổ sung cảnh báo "deprecated/superseded" vào đầu chunk cũ qua Contextual Prepend).
- **Root cause:** Kho dữ liệu tồn tại đồng thời hai phiên bản chính sách (v2023 và v2024). Bộ tìm kiếm vector và BM25 bắt trúng từ khóa "ngày phép năm" nhưng điểm số của bản cũ cao hơn do trùng khớp câu chữ mà không có bộ lọc thời gian.
- **Suggested fix:** Thêm metadata filter trong M2: chỉ truy xuất tài liệu có `status != "deprecated"`; đồng thời ở M5 Contextual Prepend, gắn nhãn rõ "[HẾT HIỆU LỰC]" vào metadata của tài liệu v2023.

---

### #2: Xung đột quy định thâm niên cộng ngày phép
- **Question:** "Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?"
- **Expected:** "Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm."
- **Got:** "Trích từ nghi_phep_nam_v2023.md. Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi 5 năm làm việc..."
- **Worst metric:** `context_recall` (0.40)
- **Error Tree:** Output sai → Context chứa văn bản lỗi thời v2023 → Thiếu thông tin từ v2024 → Reranker xếp văn bản có mật độ từ khóa cao lên trước mà không phân biệt ngày ban hành.
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Sai với quy chế hiện tại (5 năm thay vì 3 năm).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Đoạn trích dẫn không có quy chế v2024.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Cần bổ sung ngữ cảnh "theo quy định mới nhất".
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module M3 (Reranking bổ sung score penalty cho tài liệu cũ) và Module M2 (lọc theo trường version).
- **Root cause:** Bi-Encoder và BM25 cho điểm cao với cụm từ "thâm niên" trong văn bản 2023, đẩy văn bản 2024 ra khỏi top-20 ứng viên ban đầu.
- **Suggested fix:** Tăng `BM25_TOP_K` và `DENSE_TOP_K` lên 40 trước khi rerank, kết hợp trích xuất metadata `effective_date` ở M5 để ưu tiên tài liệu mới hơn.

---

### #3: Xung đột thời hạn đổi mật khẩu định kỳ (v1.0 vs v2.0)
- **Question:** "Bao lâu phải đổi mật khẩu một lần?"
- **Expected:** "Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự và thay đổi mỗi 120 ngày kèm MFA. Chính sách cũ v1.0 là 90 ngày."
- **Got:** "Trích từ mat_khau_v1.md. Mật khẩu phải được thay đổi định kỳ mỗi 90 ngày..."
- **Worst metric:** `context_precision` (0.45)
- **Error Tree:** Output chọn số liệu cũ (90 ngày) → Context ưu tiên `mat_khau_v1.md` → Do cụm từ "đổi mật khẩu một lần" trùng khớp cao với văn phong v1.
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Sai quy định bảo mật mới nhất (120 ngày + MFA).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Chỉ chứa trích đoạn v1.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Có, nên viết "Bao lâu phải đổi mật khẩu một lần theo chính sách IT hiện hành?".
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module M2 (Hybrid Search metadata filter) và Module M5 (HyQA sinh câu hỏi phân biệt phiên bản).
- **Root cause:** Từ khóa "đổi mật khẩu mỗi 90 ngày" có tần suất khớp chính xác (BM25) cao hơn câu chữ trong v2 ("chu kỳ 120 ngày").
- **Suggested fix:** Cấu hình weight của RRF hoặc dùng M5 Contextual Prepend ghi rõ: "Chính sách mật khẩu v2.0 thay thế toàn bộ quy định 90 ngày của v1.0".

---

### #4: Câu hỏi phủ định / điều kiện loại trừ
- **Question:** "Nhân viên thử việc có được nghỉ phép năm không?"
- **Expected:** "Không. Nhân viên đang trong thời gian thử việc chưa được hưởng quyền nghỉ phép năm có lương; chỉ nhân viên chính thức mới được tính ngày phép."
- **Got:** "Nhân viên chính thức được nghỉ phép năm 12 ngày (hoặc 15 ngày). Không đề cập rõ ràng điều kiện nhân viên thử việc trong đoạn trích đầu."
- **Worst metric:** `context_recall` (0.50)
- **Error Tree:** Output thiếu tính dứt khoát → Context trích xuất câu nói về nhân viên chính thức nhưng bỏ sót câu loại trừ nhân viên thử việc ở cuối văn bản → Cắt đoạn bị phân mảnh.
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Chưa trả lời thẳng vào trọng tâm phủ định ("Không được").
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Đoạn trích thiếu câu quy định loại trừ đối với nhân viên thử việc.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi rất chuẩn và tự nhiên, không cần viết lại.
  4. *Cần sửa lỗi ở module nào trong pipeline?* Module M1 (Chunking): cần tăng kích thước chunk con hoặc dùng Hierarchical Chunking để khi retrieve con thì nạp toàn bộ cha chứa cả điều khoản áp dụng và điều khoản loại trừ.
- **Root cause:** Điều khoản loại trừ nằm ở một điều mục riêng biệt, chunk con kích thước 256 ký tự đã cắt đứt mối liên kết ngữ nghĩa giữa phần quyền lợi và phần đối tượng áp dụng.
- **Suggested fix:** Trong `src/pipeline.py`, khi gửi context vào LLM, luôn lấy `parent_chunk` (2048 ký tự) thay vì chỉ lấy raw `child_chunk` (256 ký tự).

---

### #5: Câu hỏi đa bước tính toán chi phí đào tạo hoàn trả
- **Question:** "Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 6 tháng thì phải bồi hoàn bao nhiêu?"
- **Expected:** "Nhân viên cam kết làm việc tối thiểu 12 tháng. Nghỉ sau 6 tháng (còn 6 tháng cam kết) thì phải bồi hoàn 50% chi phí, tương đương 12.500.000 VNĐ."
- **Got:** "Nhân viên phải bồi hoàn chi phí đào tạo theo tỷ lệ thời gian cam kết còn lại. Không tính ra con số cụ thể 12.5 triệu."
- **Worst metric:** `answer_relevancy` (0.55) / `faithfulness`
- **Error Tree:** Context trích xuất đúng công thức tính bồi hoàn → Output chỉ trích lại công thức định tính mà không thực hiện phép tính số học cụ thể cho trường hợp câu hỏi đặt ra.
- **Phân tích 4 câu hỏi trọng tâm:**
  1. *Câu trả lời của mô hình có đúng không?* Đúng về mặt nguyên tắc nhưng chưa hoàn chỉnh vì thiếu con số cụ thể (12.500.000 VNĐ).
  2. *Các đoạn trích dẫn được đưa vào có chứa đáp án không?* Có đầy đủ công thức: chi phí hoàn trả = (Tổng chi phí / Thời gian cam kết) × Thời gian chưa làm việc.
  3. *Câu hỏi có cần viết lại cho rõ ràng hơn không?* Câu hỏi rõ ràng, có đầy đủ dữ kiện đầu vào (25 triệu, 6 tháng).
  4. *Cần sửa lỗi ở module nào trong pipeline?* LLM Synthesis Prompting (yêu cầu LLM thực hiện tính toán từng bước - Chain-of-Thought - khi gặp câu hỏi số liệu).
- **Root cause:** Prompt hệ thống yêu cầu "Trả lời CHỈ dựa trên context" quá khắt khe khiến LLM sợ ảo giác và không dám thực hiện phép tính số học suy luận từ công thức trong tài liệu.
- **Suggested fix:** Cập nhật system prompt: "Khi câu hỏi yêu cầu tính toán cụ thể dựa trên số liệu trong tài liệu, hãy trích dẫn công thức và thực hiện từng bước tính toán rõ ràng."

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
*"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*

**Error Tree walkthrough:**
1. **Output đúng?** $\rightarrow$ **SAI**. Mô hình trả lời 12 ngày phép (chính sách cũ v2023) trong khi quy định mới nhất v2024 là 15 ngày.
2. **Context đúng?** $\rightarrow$ **SAI**. Context chứa đoạn trích từ `nghi_phep_nam_v2023.md`. File `nghi_phep_nam_v2024.md` bị xếp ở rank thấp hơn và rớt khỏi top-3 sau rerank.
3. **Query rewrite OK?** $\rightarrow$ **CHƯA TỐI ƯU**. Người dùng hỏi câu hỏi ngắn, tự nhiên. Hệ thống chưa có bước Query Rewriting để bổ sung nhãn thời gian hoặc trạng thái hiện hành.
4. **Fix ở bước:**  
   - **M1:** Hierarchical chunking bảo toàn tiêu đề "Quy chế năm 2024".  
   - **M5:** Tự động trích xuất metadata `version: 2024`, `status: active` và gắn tiền tố `Contextual Prepend`.  
   - **M2:** Bổ sung metadata filtering (chỉ tìm kiếm trên tài liệu `status: active`).  
   - **M3:** Reranker được cung cấp metadata để ưu tiên phiên bản cao nhất.

**Nếu có thêm 1 giờ, sẽ optimize:**
- **1. Triển khai Temporal Metadata Filter trong Qdrant:** Tự động lọc hoặc gán trọng số suy giảm thời gian (time-decay score) cho các văn bản cũ.
- **2. Tích hợp Parent Retrieval tại Pipeline:** Thay vì truyền đoạn con 256 ký tự cho LLM, hệ thống map ngược `child.parent_id` để lấy toàn bộ đoạn cha 2048 ký tự, giải quyết triệt để vấn đề đứt đoạn điều khoản loại trừ.
- **3. Chain-of-Thought Prompting:** Cải thiện prompt trả lời để giải quyết chính xác các câu hỏi tính toán đa bước (multi-hop numeric reasoning).
