# K4-L3B — Ngày 7: Nền Tảng Dữ Liệu, Embedding & Vector Store

> Bản K4-L3B của Lab 07 (chủ đề: chính sách thương mại điện tử). Hướng dẫn Codelabs để tải lên nằm tại `../codelabs/day7-lab-data-foundations.md`; yêu cầu Giai đoạn 2 riêng xem [K4_VARIANT.md](K4_VARIANT.md). Lớp song song L3A dùng cùng bài học nhưng crawl chủ đề dịch vụ/quy định đại học.

---

## Mục Tiêu

Sau bài thực hành (lab) này, bạn cần có thể:
- Giải thích độ tương tự cosine (cosine similarity) và dự đoán điểm tương đồng giữa các văn bản
- Triển khai 3 chiến lược chia nhỏ (chunking) và so sánh ưu nhược điểm
- Xây dựng kho lưu trữ vector (vector store) với các tính năng tìm kiếm (search), lọc (filter), và xóa (delete)
- Kết nối cơ sở tri thức (knowledge base) với tác tử (agent) qua mô hình RAG
- Chỉ ra khi nào việc truy xuất (retrieval) giúp ích và khi nào nó thất bại

---

## Cấu Trúc Lab: 2 Giai Đoạn (Phase)

### Giai Đoạn 1 — Cá Nhân: Hoàn Thành gói mã nguồn `src`

Mỗi sinh viên **tự mình** hoàn thành tất cả các mục CẦN LÀM (TODO) trong `src/chunking.py`, `src/store.py`, và `src/agent.py`. Lớp dữ liệu `Document` (dataclass) và `FixedSizeChunker` đã được lập trình sẵn làm ví dụ.

### Giai Đoạn 2 — Nhóm: So Sánh Chiến Lược Truy Xuất (Retrieval Strategy)

Nhóm cùng chọn một bộ tài liệu và thống nhất 5 câu hỏi đánh giá (benchmark queries). Mỗi thành viên **thử chiến lược riêng** (chunking, metadata), chạy cùng các câu hỏi, rồi **so sánh kết quả trong nhóm** để học hỏi lẫn nhau.

---

## Thiết Lập Môi Trường

### Python 3.11 là chuẩn của Lab

Phần bắt buộc được kiểm thử trên **Python 3.11**. Dùng đúng trình thông dịch (interpreter) này khi tạo môi trường ảo (virtual environment) (`py -3.11` trên Windows hoặc `python3.11` trên macOS/Linux); file `.python-version` cũng đã khai báo phiên bản chuẩn.

```bash
pip install -r requirements.txt
pytest tests/ -v          # Phần lớn bài kiểm thử sẽ THẤT BẠI (chưa được lập trình)
```

Mặc định, lab vẫn chạy với trình nhúng giả lập `_mock_embed` nên **không bắt buộc** cài đặt mô hình nhúng (embedder) thật.
File `.env` được tự động nạp khi chạy `main.py`. Với các đoạn mã Python (snippet) chạy trực tiếp, hãy dùng lệnh `export` cho các biến môi trường cần thiết hoặc gọi hàm `load_dotenv()` nếu cần.

## Tùy Chọn Mô Hình Nhúng (Embedding Backend)

### 1) Mặc định: Trình nhúng giả lập (Mock embedder)

Không cần cài gì thêm ngoài:
```bash
pip install -r requirements.txt
```

### 2) Tùy chọn: Trình nhúng đa ngữ cục bộ (Local multilingual embedder)

```bash
pip install -r requirements-local.txt
python3 - <<'PY'
from src import LocalEmbedder
embedder = LocalEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Gói `src` hỗ trợ mô hình `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, phù hợp với kho ngữ liệu tiếng Việt, thông qua thư viện `sentence-transformers`.
- Lần chạy đầu tiên, mô hình và thư viện phụ thuộc PyTorch sẽ được tải về; đây là phần **tùy chọn**, không cần thiết để làm các TODO hoặc chạy bài kiểm thử.

### 3) Tùy chọn: Trình nhúng OpenAI (OpenAI embedder)

```bash
pip install openai
export OPENAI_API_KEY=your-key-here
python3 - <<'PY'
from src import OpenAIEmbedder
embedder = OpenAIEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Mô hình mặc định cho lựa chọn này là `text-embedding-3-small`
- Có thể đổi mô hình bằng cách:
```bash
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 4) Tùy chọn: Trình nhúng Gemini (Google Gemini embedder)

Dùng khi bạn không có OpenAI API key — Gemini API key lấy miễn phí tại [aistudio.google.com](https://aistudio.google.com/apikey), có hạn mức free tier đủ dùng cho lab.

```bash
pip install google-genai
export GEMINI_API_KEY=your-key-here
python3 - <<'PY'
from src import GeminiEmbedder
embedder = GeminiEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Mô hình mặc định cho lựa chọn này là `gemini-embedding-001`
- Có thể đổi mô hình bằng cách:
```bash
export GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

### Quy tắc dự phòng (fallback)

- Nếu không chọn gì, lab mặc định dùng `_mock_embed`
- Nếu chọn `local`, `openai`, hoặc `gemini` nhưng thiết lập bị thiếu, mã nguồn sẽ tự động chuyển về dùng `_mock_embed`
- Có thể cấu hình qua file `.env` mà không cần chạy lệnh `source .env`
- File kịch bản `main.py` chạy từ đầu đến cuối và nhập (import) các API công khai từ gói `src`

### Lệnh xác minh nhanh (verify)

Sau khi cài đặt các thư viện tùy chọn, bạn có thể kiểm tra từng backend riêng:

**Kiểm tra local embedder**

```bash
python3 - <<'PY'
from src import LocalEmbedder

embedder = LocalEmbedder()
print(embedder._backend_name, len(embedder("embedding smoke test")))
PY
```

**Kiểm tra OpenAI embedder**

```bash
python3 - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
from src import OpenAIEmbedder

load_dotenv(dotenv_path=Path(".env"), override=False)
embedder = OpenAIEmbedder()
print(embedder._backend_name, len(embedder("embedding smoke test")))
PY
```

> Lưu ý: `OpenAIEmbedder` cần biến môi trường `OPENAI_API_KEY` hợp lệ hoặc có trong file `.env`. Tương tự, `GeminiEmbedder` cần `GEMINI_API_KEY` (hoặc `GOOGLE_API_KEY`) hợp lệ.

---

## Cấu Trúc Thư Mục

```
├── README.md              ← Bạn đang đọc file này
├── exercises.md           ← Bài tập (4 phần)
├── main.py               ← Điểm bắt đầu cho việc chạy thử thủ công (manual demo)
├── bench.py              ← Kịch bản benchmark tự động 5 chiến lược + xuất JSON & HTML
├── benchmark_report.html  ← Bảng điều khiển tương tác kết quả benchmark (Tabs, Tables, Charts)
├── ket_qua_benchmark.json ← Dữ liệu benchmark chi tiết định dạng JSON
├── ket_qua_benchmark.txt  ← Tóm tắt kết quả benchmark dạng text
├── scripts/
│   ├── compare_chunking.py← So sánh 5 chiến lược trên 10 tài liệu thực tế
│   └── check_data.py      ← Kiểm tra thống kê dữ liệu đầu vào
├── src/
│   ├── chunking.py       ← 5 lớp Chunking (Fixed, Sentence, Recursive, HeaderSection, Semantic)
│   ├── store.py          ← Lớp EmbeddingStore (tìm kiếm, lọc metadata, xóa tài liệu)
│   ├── agent.py          ← Lớp KnowledgeBaseAgent
│   └── ...               ← Các module nhỏ hơn
├── data/
│   └── ecommerce/         ← 10 tài liệu chính sách Shopee tiếng Việt (~329k ký tự)
├── tests/
│   └── test_solution.py   ← Bộ kiểm thử tự động (42 tests - 100% Passed)
├── report/
│   ├── REPORT_NHOM.md    ← Báo cáo nhóm (1 file/nhóm)
│   └── REPORT_CANHAN.md  ← Báo cáo cá nhân (1 file/sinh viên)
├── docs/
│   ├── EVALUATION.md     ← Các tiêu chí đánh giá
│   ├── INSTRUCTOR_GUIDE.md ← Ghi chú dành cho giảng viên
│   └── SCORING.md        ← Tiêu chí chấm điểm
└── requirements.txt
```

---

## Các Giai Đoạn Của Lab

| Giai Đoạn | Hoạt Động |
|-----------|-----------|
| Chuẩn bị tài liệu | Nhóm chọn chủ đề, thu thập tài liệu, chuyển sang định dạng .md/.txt |
| Lập trình cá nhân | Khởi động + hoàn thành tất cả TODO (cá nhân) |
| Thiết kế chiến lược | Mỗi người thử chiến lược riêng, thống nhất 5 câu hỏi đánh giá |
| So sánh trong nhóm | Chạy đánh giá (benchmark), so sánh kết quả, chuẩn bị thuyết trình |
| Thuyết trình & thảo luận | Trình bày chiến lược + so sánh, thảo luận giữa các nhóm |

---

## Nhiệm Vụ Cá Nhân (Giai Đoạn 1)

### Đã lập trình sẵn (để tham khảo)
- `Document` dataclass — cấu trúc lưu trữ văn bản + siêu dữ liệu (metadata)
- `FixedSizeChunker` — chia nhỏ theo kích thước cố định với cơ chế cửa sổ trượt (sliding window)

### Cần lập trình (CẦN LÀM)
- `SentenceChunker` — chia nhỏ theo ranh giới câu
- `RecursiveChunker` — thử nghiệm từng dấu phân cách theo thứ tự
- `compute_similarity` — tính độ tương tự cosine
- `ChunkingStrategyComparator` — so sánh 3 chiến lược
- `EmbeddingStore` — lớp bao bọc (wrapper) cho kho lưu trữ vector (gồm 5 phương thức)
- `KnowledgeBaseAgent` — tác tử theo mô hình RAG

### Mở rộng nâng cao (Đã hoàn thành thêm)
- `HeaderSectionChunker` — chia nhỏ theo tiêu đề Markdown (`#`, `##`) và ranh giới điều khoản số (`1. `, `Bước 1:`) phù hợp với văn bản pháp lý/chính sách.
- `SemanticChunker` — chia nhỏ dựa trên độ tương tự ngữ nghĩa cosine similarity giữa các câu, tích hợp Gemini API (`google-genai`) với batching tối ưu hạn mức và fallback sang n-gram feature hashing.

---

## Nhiệm Vụ Nhóm (Giai Đoạn 2) — So Sánh Chiến Lược

1. **Chọn bộ tài liệu** (5-10 tài liệu): FAQ, Quy trình chuẩn (SOP), chính sách, tài liệu nội bộ, hoặc bất kỳ chủ đề nào
2. **Chuyển sang định dạng .txt/.md** nếu cần (xem mẹo trong exercises.md)
3. **Thống nhất 5 câu hỏi đánh giá** kèm theo câu trả lời chuẩn (gold answers)
4. **Mỗi thành viên thử chiến lược riêng**: phương pháp chunking, các tham số, cấu trúc metadata
5. **So sánh kết quả trong nhóm**: chiến lược nào cho việc truy xuất tốt hơn? Tại sao?

---

## Cách Tự Đánh Giá Kết Quả Truy Xuất (Retrieval)

Khi chạy đánh giá (benchmark), đừng chỉ hỏi **"code có chạy không?"** mà hãy tự kiểm tra 5 góc nhìn sau:

1. **Độ chính xác của truy xuất (Retrieval Precision)**
   - Top-3 kết quả trả về có chứa chunk thực sự liên quan không?
   - Điểm số (Score) có giúp phân biệt được kết quả tốt và kết quả nhiễu không?

2. **Tính mạch lạc của Chunk (Chunk Coherence)**
   - Chunk có giữ được trọn vẹn ý nghĩa không?
   - Chiến lược nào làm cho chunk dễ đọc và dễ truy xuất hơn?

3. **Tính hữu dụng của Metadata (Metadata Utility)**
   - Hàm `search_with_filter()` có giúp tăng độ chính xác không?
   - Bộ lọc có quá khắt khe, làm mất đi các kết quả tốt không?

4. **Chất lượng thông tin nền (Grounding Quality)**
   - Câu trả lời của tác tử (agent) có thực sự dựa trên ngữ cảnh được truy xuất không?
   - Bạn có thể chỉ ra chunk nào cung cấp thông tin cho câu trả lời không?

5. **Tác động của chiến lược dữ liệu (Data Strategy Impact)**
   - Bộ tài liệu mà nhóm chọn có phù hợp với các câu hỏi đánh giá không?
   - Chiến lược chunking / metadata của bạn có phù hợp với chủ đề không?

> Xem `docs/EVALUATION.md` nếu bạn muốn một danh sách kiểm tra (checklist) chi tiết hơn cho phần này.

---

## Chấm Điểm

Xem chi tiết tại `docs/SCORING.md`. Tóm tắt:

| Phần | Điểm |
|------|------|
| Cá nhân (mã nguồn + phân tích) | 60 |
| Nhóm (chiến lược + so sánh) | 40 |
| **Tổng** | **100** |

---

## Sản Phẩm Nộp Bài

1. Thư mục `src/` — hoàn thành tất cả các mục CẦN LÀM (TODO) cần thiết
2. File `report/REPORT_NHOM.md` — **một báo cáo nhóm** (chung: lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo)
3. File `report/REPORT_CANHAN.md` — **một báo cáo cá nhân cho mỗi sinh viên** (riêng: hướng tiếp cận, hoàn thiện code, dự đoán, kết quả truy xuất)

---

## Chạy Kiểm Thử

Tất cả **42/42 tests** trong `tests/test_solution.py` đã hoàn thành và vượt qua:

```bash
pytest tests/ -v
# 42 passed in ~0.07s
```

---

## Chạy Benchmark & Xem Bảng Điều Khiển Trực Quan (HTML / JSON)

Lab đã tích hợp kịch bản benchmark tự động đánh giá 5 chiến lược trên 10 tài liệu thương mại điện tử thực tế:

### 1. Chạy kịch bản Benchmark
```bash
python bench.py
```
Kịch bản sẽ:
- Đánh giá 5 chiến lược: `FixedSizeChunker`, `SentenceChunker`, `RecursiveChunker`, `HeaderSectionChunker`, `SemanticChunker`.
- Đo lường Top-1, Top-3 recall, điểm tổng kết (9/10), độ trễ (latency), và kiểm thử A/B lọc metadata.
- Xuất file kết quả JSON chi tiết: `ket_qua_benchmark.json`.
- Tạo dashboard báo cáo HTML tương tác: `benchmark_report.html` (và bản sao `ket_qua_benchmark.html`).
- Tạo file text tóm tắt: `ket_qua_benchmark.txt`.

### 2. Xem Bảng Điều Khiển Báo Cáo HTML (Interactive Dashboard)
Mở trực tiếp file `benchmark_report.html` bằng trình duyệt web:
```bash
# Trên Linux:
xdg-open benchmark_report.html
# Hoặc trên macOS:
open benchmark_report.html
# Hoặc trên Windows:
start benchmark_report.html
```
Dashboard cung cấp:
- **Tabs chuyển đổi linh hoạt**: Tổng quan so sánh, Chi tiết từng chiến lược (kèm mẫu chunks), Đánh giá 5 Query, A/B Testing Metadata, và Trình xem Raw JSON.
- **Biểu đồ so sánh trực quan**: Điểm số Benchmark và phân bố số lượng Chunks.
- **Bảng số liệu chi tiết**: Số lượng chunk, kích thước trung bình/nhỏ nhất/lớn nhất, và thời gian nạp/tìm kiếm.

### 3. So Sánh Phân Bố Chunking Nhanh
Để thống kê phân bố chunk trên toàn bộ 10 tài liệu ngữ liệu:
```bash
python scripts/compare_chunking.py
```
