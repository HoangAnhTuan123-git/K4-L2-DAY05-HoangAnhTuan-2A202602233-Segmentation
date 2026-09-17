# Báo cáo Day 5 — điền trực tiếp trong fork của bạn

**Cách dùng:** Thay mọi dấu `…` bằng bài làm thật của bạn trước khi nộp link fork trên VLearn. Giữ nguyên bốn mục và bảng để coach đọc nhanh. Viết ngắn, cụ thể theo ảnh/vùng; không cần thuật ngữ chuyên sâu. Ví dụ trong [hướng dẫn mẫu](reports/REPORT_TEMPLATE.md) chỉ giúp hiểu cách điền, không phải câu trả lời để chép lại.

- Mã học viên theo lớp: 2A202602233
- Ngày / CVAT local: 17/9/2026
- Công cụ đã dùng: Brush, Polygon, CVAT local, SegFormer & YOLO AI models

Mã học viên là mã lớp cấp; không cần ghi họ tên trong report nếu kênh VLearn đã nhận diện bạn. Chỉ ghi công cụ thật sự đã dùng; không có SAM vẫn làm bài bình thường.

## 1. Bài đã nộp

Ghi tên ZIP đúng như file trong `submissions/` và số ảnh đã vẽ, Save. Chưa làm hoặc export lỗi thì ghi `chưa có`, không tạo ZIP rỗng. Cột điểm là điểm tối đa của task, **không phải điểm tự chấm**.

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | easy_semantic.zip | 3 / 3 | 20 |
| medium_instance | medium_instance.zip | 3 / 3 | 32 |
| hard_panoptic | hard_panoptic.zip | 2 / 2 | 30 |
| cp1_holes | cp1_holes.zip | 1 / 1 | 3 |
| cp2_slice | cp2_slice.zip | 1 / 1 | 3 |
| cp5_occlusion | cp5_occlusion.zip | 1 / 1 | 3 |
| cp3_thin | cp3_thin.zip | 1 / 1 | 3 |
| cp4_curb | cp4_curb.zip | 1 / 1 | 3 |
| cp6_coverage | cp6_coverage.zip | 1 / 1 | 3 |
| **Tổng tối đa** | | | **100** |

Nếu export lỗi, ghi task, dữ liệu đã Save đến đâu và lỗi đã báo coach.

## 2. Một quyết định trước khi dùng gợi ý

Chọn object đầu tiên bạn tự vẽ ở `medium_instance`, trước khi xem bất kỳ đề xuất tự động nào cho object đó. Ghi ảnh/vị trí đủ để tìm lại; “quy tắc biên” là lý do bạn chọn hoặc dừng mask ở ranh đó.

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: `data/tiers/medium_instance/images/` ảnh 1 (`7ee6d192-89e2408b.jpg`), chiếc xe ô tô (car) màu trắng nằm ở làn đường bên phải phía trước.
- Class và quy tắc tôi dùng để chọn biên: Class `car`. Dùng Polygon bao quanh toàn bộ khung vỏ, kính xe và bánh xe chạm mặt đường; dừng biên tại mép ngoài cùng của lốp xe, không bao gồm bóng đổ (shadow) của xe in trên mặt đường.
- Nếu dùng gợi ý sau đó: vùng gợi ý sai/đúng, hành động sửa/giữ và lý do: Gợi ý tự động bị lem phần bóng đen dưới gầm xe sang mặt đường; tôi dùng Edit Polygon để co viền mask về đúng mép lốp xe tiếp xúc mặt đường vì bóng xe thuộc về lớp mặt đường `road`.
- Nếu không dùng gợi ý: không dùng

## 3. Một lỗi tôi tìm thấy và sửa

Chọn một lỗi **có thật** trong bài. Nếu công cụ lỗi khiến bạn chưa sửa được, ghi rõ đã thử gì và cần coach hỗ trợ gì; không ghi “đã sửa” khi chưa sửa.

- Task/ảnh/vùng: `easy_semantic` / ảnh `7ee6d192-89e2408b.jpg` / khu vực mép vỉa hè (`sidewalk`) giáp mặt đường (`road`).
- Lỗi thuộc loại: sai lớp / thiếu-thừa vật / gộp-tách / biên / phủ vùng / khác: biên và sai lớp (mép vỉa hè bị ăn vào lòng đường do màu nhựa đường tương đồng).
- Bằng chứng tôi nhìn thấy: Đoạn tiếp giáp giữa vỉa hè và mặt đường có màu xám đậm tương tự nhau, công cụ ban đầu vẽ lấn một dải vỉa hè sang lòng đường `road`.
- Quy tắc và hành động sửa: Dựa vào gờ bó vỉa (curbstone) nổi cao hơn mặt đường để xác định ranh giới vật lý; dùng Brush tô lại dải bị lấn thành lớp `road`.
- Sau sửa đã Save và export lại chưa? Đã Save và export lại file `easy_semantic.zip` thành công.

Nếu bạn **đã xem Summary tự đánh giá trên GitHub Actions hoặc tự chạy script**, ghi ngắn một kết quả liên quan lỗi vừa sửa (ví dụ task, metric trước/sau nếu có): `easy_semantic` đạt điểm 17.2 / 20 (IoU = 0.786 trên bản tự đánh giá). Scorecard ba tier tối đa **82**, không phải điểm cuối trên 100. Không tự ghi PASS/top 3/bonus; người phụ trách xác nhận theo tiêu chí lớp. Không đưa file ground truth vào fork.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Mỗi ca là một **vùng cụ thể** khiến bạn phải cân nhắc hai cách hiểu. Ghi dấu hiệu nhìn thấy hoặc quy tắc đã dùng, rồi nêu quyết định hoặc câu hỏi cho coach. Không cần ba lỗi; ca đã quyết định được cũng hợp lệ.

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| 1 | `cp4_curb`: Mép đá bó vỉa phân cách đường và hè | Thuộc `road` hay `sidewalk` | Đá bó vỉa nâng cao tạo gờ chắn cho người đi bộ, thuộc kết cấu hè phố | Quyết định gán toàn bộ gờ đá bó vỉa vào lớp `sidewalk`. |
| 2 | `cp1_holes`: Kính chắn gió và cửa sổ xe ô tô | Cắt rỗng (hole/void) hay giữ liền trong mask xe | Đối với object instance xe cộ, kính xe gắn liền cấu trúc thân vỏ của xe | Quyết định giữ liền toàn bộ kính cửa sổ trong mask của `car`, không khoét lỗ. |
| 3 | `medium_instance`: Người ngồi bên trong xe buýt qua lớp kính | Tách mask `person` riêng hay gộp trọn vào mask của `bus` | Người ngồi trong xe rất mờ, bị nhòe và trùng màu đen dính liền vào ghế xe, không nhìn rõ biên người | Quyết định gộp chung vào mask `bus` để tránh đoán bừa biên; hỏi coach về quy tắc tách người khi ngồi trong phương tiện. |
