# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Võ Trường An  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Dưới đây là bảng đối chiếu giữa các khái niệm lý thuyết cốt lõi trong bài giảng và hàm cụ thể đã triển khai trong mã nguồn Lab 18:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | Cắt đoạn bằng cosine similarity giữa các câu liên tiếp với mô hình `all-MiniLM-L6-v2`. Ngưỡng 0.85 giúp gom các câu cùng chủ đề thành khối mạch lạc, tránh hiện tượng cắt ngang ý hoặc ngắt giữa câu thường gặp ở fixed-size chunking. |
| **Hierarchical Chunking (Parent-Child)** | M1 | `chunk_hierarchical()` | Chia đoạn cha lớn (2048 ký tự) gắn `parent_id` và các đoạn con nhỏ (256 ký tự). Khi tìm kiếm dùng đoạn con để đạt độ chính xác từ khóa cao, khi sinh câu trả lời có thể nạp toàn bộ ngữ cảnh cha để LLM hiểu bao quát. |
| **Structure-Aware Chunking** | M1 | `chunk_structure_aware()` | Cắt đoạn dựa theo tiêu đề Markdown (`#`, `##`, `###`), giữ nguyên cấu trúc bảng và danh sách, đồng thời đưa tên đề mục vào metadata `section` của từng chunk. |
| **BM25 Tiếng Việt (Lexical Search)** | M2 | `segment_vietnamese()`, `BM25Search.index()`, `BM25Search.search()` | Xử lý tách từ ghép tiếng Việt bằng `underthesea.word_tokenize` và thay `_` bằng khoảng trắng. Điều này khắc phục triệt để lỗi BM25 tìm trượt khi người dùng nhập câu hỏi bằng từ rời. |
| **Dense Search (Vector Search)** | M2 | `DenseSearch.index()`, `DenseSearch.search()` | Sử dụng mô hình embedding đa ngữ `BAAI/bge-m3` (1024 chiều) kết hợp Qdrant Vector Database, hỗ trợ tìm kiếm ngữ nghĩa sâu và bắt từ đồng nghĩa. Cập nhật phương thức `query_points()` tương thích `qdrant-client >= 1.9`. |
| **Reciprocal Rank Fusion (RRF)** | M2 | `reciprocal_rank_fusion()` | Gộp thứ hạng từ BM25 và Dense Search theo công thức $\sum \frac{1}{k + \text{rank} + 1}$ ($k=60$). RRF giúp dung hòa thang điểm khác biệt của cosine similarity và BM25 score mà không cần chuẩn hóa thủ công. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker._load_model()`, `CrossEncoderReranker.rerank()` | Sàng lọc từ top-20 ứng viên ban đầu xuống top-3 đoạn trích đắt giá nhất bằng `BAAI/bge-reranker-v2-m3`. Khác với Bi-Encoder, Cross-Encoder chấm điểm đồng thời cả cặp `(query, doc)` để nắm bắt chi tiết từng từ khóa và quan hệ ngữ nghĩa. |
| **RAGAS 4 Metrics & Diagnostic Tree** | M4 | `evaluate_ragas()`, `failure_analysis()` | Tự động đo lường 4 chỉ số: Faithfulness, Answer Relevancy, Context Precision, Context Recall. Ánh xạ lỗi điểm thấp theo cây chẩn đoán (Diagnostic Tree) để chỉ rõ nguyên nhân và phương án khắc phục tương ứng. |
| **Contextual Prepend & Enrichment** | M5 | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | Triển khai kỹ thuật Contextual Retrieval của Anthropic: bổ sung 1 câu bối cảnh trước đoạn trích và sinh câu hỏi giả định (HyQA) để bắc cầu khoảng cách từ vựng, tối ưu chi phí qua 1 lệnh gọi LLM duy nhất. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Lỗi timeout và connection error khi tải thư viện / mô hình từ HuggingFace
- **Lỗi kỹ thuật gặp phải:**  
  `pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.`  
  và `socket.gaierror: [Errno 8] nodename nor servname provided, or not known` khi nạp mô hình qua `SentenceTransformer` do môi trường thử nghiệm hạn chế mạng ngoài.
- **Nguyên nhân gốc rễ & Cách debug:**  
  Mô hình HuggingFace mặc định luôn cố gắng gửi request kiểm tra phiên bản mới nhất trên `huggingface.co`, retry 5 lần gây chậm trễ từ 30s đến 60s dù model weights đã được lưu sẵn trong cache `~/.cache/huggingface/hub/`.
- **Cách xử lý:**  
  - Sử dụng công cụ `uv` với Python 3.11 để cài đặt dependencies siêu tốc mà không bị timeout.
  - Viết cơ chế socket probe: kiểm tra kết nối mạng tới `huggingface.co` trong 1 giây; nếu không kết nối được thì tự động bật cờ `os.environ["HF_HUB_OFFLINE"] = "1"` để nạp ngay từ local snapshot trong 0.1 giây.

### 2. Sự cố xung đột tokenization tiếng Việt trong BM25
- **Lỗi kỹ thuật gặp phải:**  
  BM25 trả về điểm 0 cho hầu hết các câu hỏi chứa từ ghép (ví dụ: "nghỉ phép", "thâm niên").
- **Nguyên nhân gốc rễ:**  
  `underthesea` khi tokenize tạo từ ghép với dấu gạch dưới `nghỉ_phép`. Trong khi đó người dùng gõ `nghỉ phép` (phân tách bằng khoảng trắng). BM25 coi đây là hai token khác nhau hoàn toàn.
- **Cách xử lý:**  
  Trong hàm `segment_vietnamese()`, thực hiện `.replace("_", " ")` để đưa mọi từ ghép về định dạng khoảng trắng đồng nhất, giúp BM25 khớp chính xác từng từ.

### 3. Phương thức `search()` bị deprecate trên `qdrant-client` mới
- **Lỗi kỹ thuật gặp phải:**  
  `AttributeError: 'QdrantClient' object has no attribute 'search'` hoặc cảnh báo deprecation khi dùng `recreate_collection`.
- **Nguyên nhân gốc rễ:**  
  `qdrant-client >= 1.9` đã chuyển sang chuẩn `query_points()` và khuyến nghị dùng `collection_exists` + `create_collection`.
- **Cách xử lý:**  
  Đổi cú pháp truy vấn sang `self.client.query_points(collection, query=query_vector, limit=top_k)` và kiểm tra tồn tại collection trước khi tạo mới.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Hệ thống Trợ lý ảo Tra cứu Quy chế & Quy trình Vận hành Nội bộ Doanh nghiệp (Enterprise SOP & Policy RAG Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với CharacterTextSplitter (chunk_size=1000, overlap=100) và Dense Search đơn lẻ trên OpenAI `text-embedding-3-small`.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Context Precision thấp:* Thường trích xuất lẫn lộn văn bản cũ và văn bản mới khi doanh nghiệp cập nhật quy chế hàng năm.
  - *Bỏ sót điều khoản loại trừ:* Chunk cố định thường cắt ngang giữa điều khoản cho phép và điều khoản loại trừ, dẫn đến mô hình trả lời sai quyền lợi của nhân viên thử việc/nghỉ việc.
  - *Tốc độ và chi phí:* Prompt quá dài do nhồi nhét nhiều chunk không liên quan làm tăng chi phí token và tăng độ trễ trả lời.

#### 2. Kế hoạch cải tiến
1. **Chunking Strategy:**  
   Chuyển sang **Hierarchical Chunking (Parent-Child)** kết hợp **Structure-Aware**. Với tài liệu quy chế (SOP), dùng tiêu đề Markdown để định hình đoạn cha (2048 ký tự), sau đó cắt nhỏ thành các đoạn con (256 ký tự). Retrieve trên đoạn con nhưng inject toàn bộ đoạn cha vào prompt LLM.
2. **Search Retrieval:**  
   Triển khai **Hybrid Search (BM25 tiếng Việt + Dense BAAI/bge-m3)** hợp nhất qua **RRF ($k=60$)**. BM25 đảm bảo bắt chính xác các mã số hiệu văn bản, số tiền, ngày phép; BAAI/bge-m3 hỗ trợ hiểu các câu hỏi diễn đạt tự nhiên theo nghĩa tương đương.
3. **Reranking:**  
   Tích hợp tầng Cross-Encoder **`BAAI/bge-reranker-v2-m3`** lọc từ top-20 xuống top-3 đoạn trích có độ liên quan cao nhất trước khi gửi tới LLM.
4. **Enrichment:**  
   Áp dụng **Contextual Prepend** của Anthropic: tự động trích xuất metadata `version`, `effective_date`, `status` và đính kèm câu mô tả nguồn vào đầu mỗi chunk.
5. **Evaluation:**  
   Xây dựng bộ benchmark 50 câu hỏi nội bộ và tự động đánh giá định kỳ bằng **RAGAS 4 metrics**, kết hợp Diagnostic Tree để phân loại lỗi hồi quy (regression testing).

#### 3. Timeline triển khai
- **Tuần 1:** Chuẩn hóa dữ liệu SOP sang định dạng Markdown cấu trúc, áp dụng Module M1 (Hierarchical Chunking) và M5 (Contextual Prepend kèm trích xuất metadata phiên bản).
- **Tuần 2:** Dựng cụm Qdrant trên server nội bộ, thiết lập Hybrid Search BM25 + Qdrant BGE-M3 (Module M2).
- **Tuần 3:** Tích hợp tầng Cross-Encoder Reranker M3 và tối ưu độ trễ với ONNX/FlashRank cho môi trường CPU/GPU.
- **Tuần 4:** Thiết lập luồng tự động đánh giá RAGAS (Module M4) trong quy trình CI/CD và triển khai thử nghiệm trên nhóm người dùng nội bộ.
