# Ngày 7 — Bài tập
## Nền tảng Dữ liệu: Embedding & Vector Store | Bài tập thực hành

---

## Phần 1 — Khởi động (Cá nhân)

### Bài tập 1.1 — Cosine Similarity (Độ tương tự Cosine) bằng ngôn ngữ đời thường

Không yêu cầu toán học — giải thích về mặt khái niệm:

- **Điều gì xảy ra khi hai đoạn văn bản có độ tương tự cosine cao?**
  > Khi hai đoạn văn bản có độ tương tự cosine cao (tiến gần đến 1.0), hướng của hai vector embedding đại diện cho chúng trong không gian đa chiều gần như trùng nhau (góc $\theta$ giữa hai vector xấp xỉ $0^\circ$). Về mặt ngữ nghĩa, điều này có nghĩa là hai văn bản đề cập đến cùng một chủ đề, chia sẻ các khái niệm, ý định (intent) hoặc thông tin cốt lõi tương đồng, ngay cả khi chúng sử dụng các từ ngữ diễn đạt khác nhau.

- **Ví dụ cụ thể về hai câu có độ tương tự CAO và hai câu có độ tương tự THẤP:**
  - **Độ tương tự CAO:**
    - *Câu A:* "Chính sách đổi trả hàng hóa trên sàn thương mại điện tử Shopee."
    - *Câu B:* "Quy định và thủ tục hoàn tiền cho người mua khi sản phẩm giao bị lỗi."
    - *Tại sao tương đồng:* Cả hai câu cùng tập trung vào nghiệp vụ hậu mãi thương mại điện tử (bảo vệ quyền lợi người mua khi phát sinh sự cố đơn hàng). Điểm thực tế (Gemini Embeddings): **0.7426**.
  - **Độ tương tự THẤP:**
    - *Câu A:* "Bảo mật thông tin tài khoản ngân hàng và mật khẩu người dùng."
    - *Câu B:* "Thời gian giao hàng tiêu chuẩn của đơn vị vận chuyển Viettel Post."
    - *Tại sao khác biệt:* Câu A nói về an toàn thông tin và quyền riêng tư, câu B nói về thời gian vận chuyển logistics. Điểm thực tế: **0.5277**.

- **Tại sao độ tương tự cosine lại được ưu tiên hơn khoảng cách Euclid (Euclidean distance) đối với text embeddings?**
  > Khoảng cách Euclid chịu ảnh hưởng mạnh bởi độ dài tuyệt đối (magnitude/norm) của vector. Một đoạn văn bản dài có xu hướng tích lũy vector lớn hơn một câu ngắn, khiến khoảng cách Euclid giữa chúng bị kéo xa dù cùng bàn về một chủ đề. Ngược lại, Cosine similarity chỉ đo góc giữa các vector và triệt tiêu ảnh hưởng của độ dài vector ($\frac{A \cdot B}{\|A\| \|B\|}$), giúp so sánh ngữ nghĩa công bằng giữa các văn bản có kích thước khác nhau (scale-invariant).

---

### Bài tập 1.2 — Bài toán tính toán Chunking

- **Một tài liệu có độ dài 10,000 ký tự. Bạn tiến hành chia nhỏ (chunk) với `chunk_size=500` (kích thước chunk), `overlap=50` (độ chồng chéo). Bạn dự kiến sẽ có bao nhiêu chunks?**
  - Công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
  - Bước nhảy (step) giữa các chunk: `500 - 50 = 450` ký tự.
  - Áp dụng công thức:
    $$\text{Số lượng chunk} = \left\lceil \frac{10000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = \lceil 22.11 \rceil = 23 \text{ chunks}$$
  - *Đáp án:* **23 chunks**.

- **Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk sẽ thay đổi như thế nào? Tại sao bạn lại muốn tăng độ chồng chéo?**
  - Bước nhảy mới: `500 - 100 = 400` ký tự.
  - Số lượng chunk mới:
    $$\text{Số lượng chunk} = \left\lceil \frac{10000 - 100}{500 - 100} \right\rceil = \left\lceil \frac{9900}{400} \right\rceil = \lceil 24.75 \rceil = 25 \text{ chunks}$$
  - *Thay đổi:* Tăng từ 23 lên **25 chunks** (tăng 2 chunks, tương đương +8.7%).
  - *Tại sao muốn tăng độ chồng chéo:*
    1. **Bảo toàn ngữ cảnh ranh giới (cross-boundary context):** Tránh hiện tượng một câu văn quan trọng, một điều kiện loại trừ, hoặc một định nghĩa pháp lý bị xé đôi giữa hai chunk liên tiếp.
    2. **Tăng xác suất truy xuất đúng thực thể (entity preservation):** Khi câu hỏi truy vấn tìm kiếm một cụm thông tin nằm sát biên cắt, overlap đủ lớn đảm bảo ít nhất một trong hai chunk chứa đầy đủ ngữ cảnh để LLM sinh câu trả lời chính xác.

---

## Phần 2 — Lập trình cốt lõi (Cá nhân)

Tất cả các thành phần trong gói mã nguồn `src/` đã được lập trình hoàn chỉnh và vượt qua toàn bộ 42 bài kiểm thử tự động (`pytest tests/ -v`).

### Danh sách cần làm (Checklist)
- [x] `Document` dataclass — ĐÃ TRIỂN KHAI SẴN
- [x] `FixedSizeChunker` — ĐÃ TRIỂN KHAI SẴN
- [x] `SentenceChunker` — tách dựa trên ranh giới câu (`re.split(r"(?<=[.!?])\s+", text)`), nhóm $N$ câu thành 1 chunk.
- [x] `RecursiveChunker` — đệ quy tách theo mức ưu tiên phân cách `["\n\n", "\n", ". ", " ", ""]` và ghép lại không vượt `chunk_size`.
- [x] `compute_similarity` — tính cosine similarity: $\frac{a \cdot b}{\|a\| \|b\|}$, xử lý an toàn vector rỗng hoặc chuẩn bằng 0.
- [x] `ChunkingStrategyComparator` — so sánh các chiến lược chunking trên cùng văn bản, tính số lượng, độ dài trung bình, min, max.
- [x] `EmbeddingStore.__init__` — khởi tạo kho vector trong bộ nhớ với collection và hàm embedding.
- [x] `EmbeddingStore.add_documents` — nhúng (embed) và lưu trữ từng tài liệu cùng metadata.
- [x] `EmbeddingStore.search` — nhúng query, tính tích vô hướng (dot product) và sắp xếp giảm dần theo điểm số top-$k$.
- [x] `EmbeddingStore.get_collection_size` — trả về số lượng documents/chunks hiện có trong kho.
- [x] `EmbeddingStore.search_with_filter` — lọc chính xác theo metadata trước (pre-filtering), sau đó xếp hạng similarity.
- [x] `EmbeddingStore.delete_document` — xóa toàn bộ các bản ghi theo `doc_id`.
- [x] `KnowledgeBaseAgent.answer` — truy xuất top-$k$ ngữ cảnh + format numbered context prompt + gọi hàm LLM.

---

## Phần 3 — So Sánh Chiến Lược Truy Xuất (Nhóm)

### Bài tập 3.0 — Chuẩn Bị Tài Liệu

**Chủ đề:** Chính sách và quy định pháp lý của sàn thương mại điện tử Shopee Việt Nam (`help.shopee.vn`).

#### Bảng danh mục tài liệu (10 tài liệu trong `data/ecommerce/`):

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Điều khoản dịch vụ | https://help.shopee.vn/portal/4/article/77243 | 20/09/2026 / v4 | 83.367 | `audience: buyer`, `category: terms-of-service` |
| 2 | Chính sách bảo mật | https://help.shopee.vn/portal/4/article/77244 | 20/09/2026 / v4 | 43.105 | `audience: buyer`, `category: privacy-policy` |
| 3 | Quy chế hoạt động sàn TMĐT Shopee.vn | https://help.shopee.vn/portal/4/article/77245 | 20/09/2026 / v4 | 77.868 | `audience: buyer`, `category: platform-rules` |
| 4 | Quy định về đăng bán sản phẩm trên Shopee | https://help.shopee.vn/portal/4/article/77246 | 20/09/2026 / v4 | 21.532 | `audience: seller`, `category: listing-policy` |
| 5 | Chính sách cấm/hạn chế sản phẩm | https://help.shopee.vn/portal/4/article/77247 | 20/09/2026 / v4 | 12.850 | `audience: seller`, `category: prohibited-products-policy` |
| 6 | Chính sách vận chuyển Shopee | https://help.shopee.vn/portal/4/article/77250 | 20/09/2026 / v4 | 24.597 | `audience: buyer`, `category: shipping-policy` |
| 7 | Chính sách trả hàng và hoàn tiền | https://help.shopee.vn/portal/4/article/77251 | 20/09/2026 / v4 | 19.609 | `audience: buyer`, `category: returns-policy` |
| 8 | Điều khoản sử dụng dịch vụ hiển thị | https://help.shopee.vn/portal/4/article/77252 | 20/09/2026 / v4 | 7.630 | `audience: seller`, `category: ads-display-policy` |
| 9 | Điều khoản dịch vụ của Shopee Mall | https://help.shopee.vn/portal/4/article/77262 | 20/09/2026 / v4 | 33.736 | `audience: buyer`, `category: terms-of-service` |
| 10 | Quy trình giải quyết tranh chấp/xử lý khiếu nại | https://help.shopee.vn/portal/4/article/77265 | 20/09/2026 / v4 | 4.838 | `audience: buyer`, `category: dispute-resolution` |

---

### Bài tập 3.1 — Thiết Kế Chiến Lược Truy Xuất (Mỗi người thử riêng)

#### 1. Đường cơ sở (Baseline):
Chạy lệnh `.venv/bin/python scripts/compare_chunking.py` trên 3 tài liệu tiêu biểu:

| Tài liệu | Chiến lược (Strategy) | Số Chunks | Độ dài TB | Min | Max | Nhận xét tính toàn vẹn |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **chinh-sach-bao-mat.md** (43.105 kt) | FixedSizeChunker | 96 | 498.5 | 355 | 500 | Cắt ngang câu/đoạn, phụ thuộc overlap |
| | SentenceChunker | 65 | 659.2 | 75 | 3747 | Giữ trọn câu nhưng chunk cực dài (3.747 kt) |
| | RecursiveChunker | 111 | 385.7 | 72 | 500 | Phân đoạn tự nhiên, kiểm soát trần <= 500 |
| | HeaderSectionChunker | 117 | 365.5 | 5 | 500 | Giữ trọn cấu trúc điều khoản theo số mục |
| | SemanticChunker | 192 | 222.5 | 2 | 2626 | Gom ý theo ngữ nghĩa, chuyển cụm khi đổi chủ đề |
| **chinh-sach-tra-hang-hoan-tien.md** (19.609 kt) | FixedSizeChunker | 44 | 494.5 | 259 | 500 | Kích thước đều nhưng mất ranh giới mục |
| | SentenceChunker | 48 | 405.6 | 24 | 984 | Giữ câu tốt, kích thước biến thiên |
| | RecursiveChunker | 54 | 361.1 | 170 | 500 | Giữ được ngữ cảnh mục điều khoản đổi trả |
| | HeaderSectionChunker | 55 | 354.1 | 110 | 495 | Tách đúng từng trường hợp trả hàng |
| | SemanticChunker | 141 | 137.4 | 2 | 847 | Phát hiện điểm ngắt khi chuyển đối tượng |
| **dieu-khoan-dich-vu.md** (83.367 kt) | FixedSizeChunker | 186 | 497.9 | 117 | 500 | Điều khoản dài bị ngắt làm 2-3 phần |
| | SentenceChunker | 161 | 515.0 | 24 | 3659 | Câu ghép pháp luật quá dài, vượt quota |
| | RecursiveChunker | 222 | 373.4 | 67 | 500 | Ổn định nhất cho văn bản quy phạm pháp lý |
| | HeaderSectionChunker | 228 | 363.2 | 12 | 500 | Tách chuẩn xác theo 29 điều khoản lớn |
| | SemanticChunker | 480 | 172.1 | 2 | 3026 | Tách chi tiết theo từng khía cạnh nghĩa vụ |

#### 2. Thiết kế chiến lược nâng cao:
Nhóm đã mở rộng và triển khai 2 chiến lược chuyên sâu trong `src/chunking.py`:
- **`HeaderSectionChunker`**: Tách tài liệu theo các tiêu đề Markdown (`#`, `##`, `###`) và các điều khoản đánh số (`1. Quy định...`, `Bước 1:`). Bảo tồn tính nguyên tử của từng điều khoản pháp lý.
- **`SemanticChunker` (hỗ trợ Gemini API)**: Sử dụng mô hình embedding (`gemini-embedding-001` qua `google-genai`) để tính độ tương đồng giữa các câu liên tiếp, tự động đặt ranh giới chunk tại điểm sụt giảm ngữ nghĩa (semantic shift < threshold).

```python
# Trích từ src/chunking.py - SemanticChunker
class SemanticChunker:
    def __init__(self, similarity_threshold: float = 0.65, max_chunk_size: int = 800, use_gemini: bool = False):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.use_gemini = use_gemini
    
    def chunk(self, text: str) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        embeddings = self._embed_sentences(sentences) # Batch embed Gemini or smart mock
        chunks, current = [], [sentences[0]]
        for i in range(len(sentences) - 1):
            sim = compute_similarity(embeddings[i], embeddings[i + 1])
            if (sim < self.similarity_threshold or len(" ".join(current)) + len(sentences[i+1]) > self.max_chunk_size):
                chunks.append(" ".join(current).strip())
                current = [sentences[i + 1]]
            else:
                current.append(sentences[i + 1])
        if current: chunks.append(" ".join(current).strip())
        return chunks
```

---

### Bài tập 3.2 — Chuẩn Bị Câu Hỏi Đánh Giá (Benchmark Queries)

Bộ 5 câu hỏi chuẩn hóa thống nhất cho nhóm:

| # | Câu hỏi (Query) | Loại câu hỏi | Câu trả lời chuẩn (Gold Answer) | Tài liệu đích |
|---|----------------|--------------|-------------------------------|---------------|
| 1 | Thời hạn giải quyết các tranh chấp không phải là Trả Hàng/Hoàn Tiền là bao nhiêu ngày làm việc kể từ khi nhận đủ tài liệu? | Tra số liệu | Trong vòng 07 ngày làm việc kể từ ngày nhận được đầy đủ các thông tin/tài liệu có liên quan đến vụ việc. | `quy-trinh-giai-quyet-tranh-chap` (Bước 3) |
| 2 | Điều kiện về tỷ lệ và thời hạn sử dụng tối thiểu đối với hàng hóa khi người bán giao đi là gì? | Hỏi điều kiện | Người Bán chỉ được phép bán các loại hàng hóa mà khi giao đi phải còn ít nhất 30% thời hạn sử dụng và còn ít nhất 30 ngày, tính từ thời điểm hiện tại đến ngày hết hạn. | `quy-dinh-dang-ban-san-pham` (Mục 3.a) |
| 3 | Quy trình giải quyết tranh chấp hoặc xử lý khiếu nại trên sàn Shopee gồm các bước nào? | Hỏi quy trình | Gồm 4 bước: Bước 1: Khiếu nại qua ứng dụng Shopee (Đơn Mua); Bước 2: Shopee tiếp nhận và xác minh; Bước 3: Xử lý theo Chính sách hoặc trong 07 ngày làm việc; Bước 4: Chuyển cơ quan nhà nước nếu vượt thẩm quyền. | `quy-trinh-giai-quyet-tranh-chap` (Mục 1) |
| 4 | Liệt kê các trường hợp đơn vị vận chuyển có quyền từ chối tiếp nhận vận chuyển kiện hàng? | Liệt kê | Khi hàng thuộc danh mục hàng cấm/nguy hiểm có nguy cơ cháy nổ; kiện hàng không đóng gói đúng quy chuẩn; vượt quá kích thước hoặc khối lượng tối đa; hoặc thông tin chênh lệch thực tế. | `chinh-sach-van-chuyen` (Mục 2) |
| 5 | Quy định và thủ tục xử lý khi phát sinh yêu cầu đổi trả đối với sản phẩm giao sai hoặc hư hỏng là gì? | Cần metadata_filter | Người Mua có quyền gửi yêu cầu Trả hàng/Hoàn tiền trên ứng dụng Shopee khi nhận hàng sai hoặc hư hại, cần cung cấp bằng chứng (hình ảnh/video mở gói hàng) và gửi trả hàng nguyên vẹn theo hướng dẫn. (*Lọc audience=buyer*) | `chinh-sach-tra-hang-hoan-tien` |

---

### Bài tập 3.3 — Dự Đoán Độ Tương Tự Cosine (Cá nhân)

Kiểm thử hàm `compute_similarity()` trên 5 cặp câu thực tế:

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (Gemini) | Điểm thực tế (Mock) | Đánh giá |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | Chính sách đổi trả hàng hóa trên sàn thương mại điện tử Shopee. | Quy định và thủ tục hoàn tiền cho người mua khi hàng bị lỗi. | Cao | **0.7426** | -0.2561 | Đúng dự đoán với Gemini |
| 2 | Người bán phải bảo đảm hàng hóa còn hạn sử dụng tối thiểu 30 ngày. | Điều kiện thời hạn sử dụng sản phẩm khi giao cho đơn vị vận chuyển. | Rất Cao | **0.7941** | 0.0732 | Đúng dự đoán với Gemini |
| 3 | Đơn vị vận chuyển có quyền từ chối tiếp nhận các kiện hàng cấm. | Danh mục hàng hóa nguy hiểm bị cấm vận chuyển theo quy định pháp luật. | Rất Cao | **0.7998** | -0.1312 | Đúng dự đoán với Gemini |
| 4 | Quy trình khiếu nại và giải quyết tranh chấp qua ứng dụng Shopee. | Chương trình khuyến mãi giảm giá ngày hội mua sắm 9.9. | Thấp | **0.5482** | 0.1040 | Đúng dự đoán (sụt giảm điểm rõ rệt) |
| 5 | Bảo mật thông tin tài khoản ngân hàng và mật khẩu người dùng. | Thời gian giao hàng tiêu chuẩn của đơn vị vận chuyển Viettel Post. | Rất Thấp | **0.5277** | -0.2508 | Đúng dự đoán (hai miền ý nghĩa rời rạc) |

**Nhận xét sâu sắc:**
- **Mô hình nhúng thực (Gemini Embeddings)** phản ánh xuất sắc quan hệ ngữ nghĩa: các cặp câu cùng chủ đề đạt điểm từ $0.74$ đến $0.80$, trong khi các cặp khác chủ đề giảm sâu xuống $0.52 - 0.54$.
- **MockEmbedder** thuần túy băm chuỗi ra số giả ngẫu nhiên, cho ra các giá trị dao động quanh 0 mà không có tính logic ngữ nghĩa.

---

### Bài tập 3.4 — Chạy Đánh Giá & So Sánh Trong Nhóm

Kết quả thực thi từ script benchmark hoàn chỉnh (`.venv/bin/python bench.py`):

| Chiến lược (Strategy) | Số Chunks | Độ dài TB | Min - Max | Top-3 Recall | Tổng Điểm (/10) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FixedSizeChunker** (`fixed_size`) | 734 | 497.7 ký tự | 117 – 500 kt | 5/5 (100.0%) | **9/10** |
| **SentenceChunker** (`sentence`) | 759 | 430.4 ký tự | 24 – 3747 kt | 5/5 (100.0%) | **9/10** |
| **RecursiveChunker** (`recursive`) | 856 | 382.1 ký tự | 16 – 500 kt | 4/5 (80.0%) | **9/10** |
| **HeaderSectionChunker** (`header_section`) | 789 | 360.5 ký tự | 12 – 500 kt | 5/5 (100.0%) | **9/10** |
| **SemanticChunker** (`semantic`) | 2243 | 185.3 ký tự | 2 – 800 kt | 4/5 (80.0%) | **9/10** |

**Chiến lược nào tốt nhất cho chủ đề chính sách sàn thương mại điện tử?**
> **`HeaderSectionChunker` và `RecursiveChunker`** là sự lựa chọn tối ưu nhất trong sản xuất (production). Mặc dù `FixedSize` đạt điểm cao về mặt thống kê từ khóa, nhưng nó xé ngang các điều khoản pháp lý. `HeaderSectionChunker` bám sát các điều khoản đánh số (`1.`, `Bước 1:`, `#`), vừa đảm bảo tính trọn vẹn của từng điều khoản, vừa kiểm soát độ dài không vượt ngưỡng token của LLM.

---

### Bài tập 3.5 — Phân Tích Lỗi (Failure Analysis)

- **Trường hợp lỗi:** Câu hỏi #5 khi không có bộ lọc metadata (`filter=None`).
- **Hiện tượng:** Truy vấn: *"Quy định và thủ tục xử lý khi phát sinh yêu cầu đổi trả đối với sản phẩm giao sai hoặc hư hỏng là gì?"*
  - Khi **CÓ FILTER** (`audience: buyer`): Top-1 và Top-2 trả về đúng `chinh-sach-tra-hang-hoan-tien` và `dieu-khoan-dich-vu-shopee-mall` (chính sách đổi trả của Người Mua).
  - Khi **KHÔNG CÓ FILTER**: Hệ thống nhặt các đoạn văn bản từ `quy-dinh-dang-ban-san-pham` và `chinh-sach-cam-han-che-san-pham` (dành cho Người Bán). Do người bán cũng có quy định về hàng hóa hư hại và trách nhiệm bồi thường, các chunk này có độ tương đồng từ khóa cao, khiến Agent trích dẫn nghĩa vụ của Người Bán thay vì quyền khiếu nại của Người Mua.
- **Nguyên nhân cốt lõi:** Câu hỏi mang tính tổng quát, không nêu rõ danh xưng đối tượng. Kho dữ liệu thương mại điện tử có hai persona đối kháng (`buyer` vs `seller`).
- **Đề xuất cải tiến:**
  1. Luôn áp dụng tiền lọc metadata (pre-filtering) theo vai trò phiên đăng nhập của người dùng.
  2. Bổ sung trường metadata `persona` hoặc `sub_category` trong YAML frontmatter để phân tách rạch ròi luồng giải quyết khiếu nại.

---

## Danh Sách Kiểm Tra Nộp Bài (Submission Checklist)

- [x] Vượt qua tất cả các bài kiểm thử (tests): `pytest tests/ -v` (42/42 passed)
- [x] Cập nhật thư mục `src/` (cá nhân) với đầy đủ các chunkers, store và agent.
- [x] Hoàn thành báo cáo nhóm (`report/REPORT_NHOM.md` — 1 file/nhóm)
- [x] Hoàn thành báo cáo cá nhân (`report/REPORT_CANHAN.md` — 1 file/sinh viên)
- [x] Hoàn thiện kết quả benchmark JSON (`ket_qua_benchmark.json`) và HTML dashboard (`benchmark_report.html`)
