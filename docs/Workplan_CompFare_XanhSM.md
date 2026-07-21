# WORK PLAN — COMPFARE-XANHSM

**Competitor Fare Forecasting with Calibrated Uncertainty**

> File tạm / bản nháp kế hoạch. Đây là tài liệu lập kế hoạch để điền form, tách riêng khỏi technical report của dự án. Mỗi sprint = 2 tuần, tổng 6 tuần: Sprint 1 = tuần 1–2, Sprint 2 = tuần 3–4, Sprint 3 = tuần 5–6.

## Đăng ký project

### Tên Project
CompFare-XanhSM: Competitor Fare Forecasting with Calibrated Uncertainty

### Sản phẩm bàn giao
PoC ước lượng giá đối thủ (Grab/Be) tại thời điểm hiện tại từ quan sát có độ trễ, gồm ba cấu phần:
(i) phân tích yếu tố quyết định price và price multiplier;
(ii) model dự báo price và multiplier từ delayed observations;
(iii) uncertainty quantification xuất prediction interval đã hiệu chuẩn (ví dụ 80% PI: [17.20, 20.60]).
Kèm: benchmark trên public data (Uber/Lyft Boston, 638k bản ghi, 23 ngày, có weather), decision rule cho pricing algo, source code, technical report và 6 weekly report.

### Tiêu chí hoàn thành (kết quả đầu ra mong muốn)
• Pipeline chạy end-to-end từ raw data đến prediction interval, tái lập được.
• (i) Xác định và lượng hóa được các driver chính của price và multiplier.
• (ii) Cải thiện tối thiểu 20% MAE so với baseline persistence, trên time-based split không leakage.
• (iii) Coverage của 80% và 90% PI sai lệch ≤3 điểm phần trăm so với mức danh nghĩa, với interval width nhỏ nhất có thể.
• Interval nới đúng chỗ: coverage được kiểm theo lát cắt (cao điểm, thời tiết, surge, route hiếm, độ trễ).
• Có decision rule: ngưỡng interval mà pricing algo được phép dùng tín hiệu giá đối thủ.
• Mỗi tuần một weekly report với punchline đóng vai trò sprint goal.

### Công việc Sprint 1
Trọng tâm: cấu phần (i) và baseline của (ii).
• Chốt data contract: delay τ, đơn vị dự báo (route × service × time bucket), target gồm price và multiplier.
• Rebuild snapshot ở bucket 5–10 phút thay bản theo giờ; giữ observation age làm feature; lọc route đủ dày.
• EDA (i): Hiếu phân tích driver của giá; Chiến phân tích driver của biến động giá.
• Baseline persistence + model gradient boosting đầu tiên; chốt protocol đánh giá và kiểm tra leakage.
• Khảo sát public dataset bổ sung; xin mentor distribution hoặc simulated data.

### Công việc Sprint 2
Trọng tâm: hoàn thiện (ii), khởi động (iii).
• Feature engineering: lag price, rolling mean/std, route encoding, weather, observation age.
• So sánh có kiểm soát các model trên cùng test set; thử ensemble nếu sai số bổ trợ nhau.
• Ablation weather / lịch sử giá / route; phân tích sai số theo τ và theo lát cắt.
• EDA (iii): yếu tố quyết định độ bất định của dự báo.
• Prototype uncertainty: quantile regression, split conformal, ensemble interval; đo coverage sơ bộ.

### Công việc Sprint 3
Trọng tâm: hoàn thiện (iii) và bàn giao.
• Adaptive/Mondrian conformal: hiệu chuẩn theo nhóm để interval phản ánh đúng mức khó.
• Đánh giá đầy đủ: marginal và conditional coverage, width, pinball loss, CRPS, reliability diagram.
• Stress test nhiều kịch bản; xây decision rule và mô phỏng tác động; đo decision latency.
• Hoàn thiện PoC, report, demo và gói bàn giao để mentor test trên dataset GreenSM.

## Ghi chú

**Cấu trúc project theo định hướng mentor:** (i) study quan hệ giữa các key feature với price và price multiplier; (ii) build model dự báo price và price multiplier từ delayed observations; (iii) uncertainty quantification để xuất prediction interval đã hiệu chuẩn.

**Phân công:** Hiếu focus cấu phần (ii), trọng tâm EDA là “What features determine price?”. Chiến focus cấu phần (iii), trọng tâm EDA là “What determines forecast uncertainty?”. Sprint 1–2 Chiến support Hiếu ở phần model vì (ii) phải xong trước mới roll out được (iii); Sprint 3 Hiếu support Chiến ở phần uncertainty vì đây là phần khó và dễ mất thời gian nhất. Phân công là trọng số, không phải ranh giới cứng.

**Chế độ báo cáo:** nộp weekly report vào cuối mỗi tuần, không nộp daily. Mỗi report mở đầu bằng (a) vấn đề đang giải quyết hoặc weakness của tuần trước, và (b) punchline — key contribution của tuần, đóng vai trò sprint goal.

**KPI và phạm vi:** Ngưỡng cải thiện MAE ≥20% so với baseline persistence và sai lệch coverage ≤3 điểm phần trăm là KPI đề xuất, cần hiệu chỉnh sau Sprint 1 dựa trên baseline thực tế. Eval thực hiện trên public dataset (Uber/Lyft Boston và các public dataset khác nếu tìm được). Phần test trên dữ liệu thật do mentor thực hiện và không nằm trong scope cam kết — nhóm chịu trách nhiệm bàn giao model ở trạng thái chạy được trên dữ liệu khác. Dataset chỉ có 23 ngày là hạn chế đã biết: mọi claim về seasonality dài hạn nằm ngoài phạm vi kết luận.

## Bảng kế hoạch chi tiết 6 tuần

| Tuần | Mục tiêu và punchline dự kiến | Công việc chính | Sản phẩm bàn giao | Tiêu chí hoàn thành và cổng kiểm soát |
|---|---|---|---|---|
| 1<br>(Sprint 1) | Punchline: Giá đối thủ do những yếu tố nào quyết định, và quan sát trễ bao lâu thì còn dùng được.<br>• Hoàn thành cấu phần (i).<br>• Khóa đầu vào để phần sau không phải làm lại. | • Chốt data contract: τ, đơn vị dự báo, target price và multiplier. (cả nhóm)<br>• Đo phân bố khoảng cách giữa hai quan sát liên tiếp để chốt τ khả thi. Sơ bộ: median 16,6 phút; 35% gap ≤ 10 phút. (Hiếu)<br>• Rebuild snapshot ở bucket 5–10 phút, giữ observation age làm feature; lọc route đủ dày. (Hiếu)<br>• EDA (i) — Hiếu: price và multiplier theo giờ, thứ, quãng đường, route, loại dịch vụ, thời tiết.<br>• EDA (i) — Chiến: biến động giá theo bước 5–10 phút và autocorrelation, làm tiền đề cho (iii).<br>• Khảo sát public dataset bổ sung; xin mentor distribution/simulated data. (Chiến) | • week1_report.md<br>• data_contract.md<br>• snapshot_table_10min<br>• eda_price_drivers.md<br>• price_volatility_analysis.md<br>• dataset_survey.md | • Định nghĩa τ, đơn vị dự báo và target được mentor review.<br>• Snapshot không chứa thông tin tương lai, có script kiểm tra leakage.<br>• EDA nêu tối thiểu 3 phát hiện định lượng về driver của giá.<br>• Gate: nếu tập route đủ dày quá nhỏ, nới bucket lên 15 phút và ghi rõ khoảng cách so với đề bài.<br>• Gate: nếu persistence đã gần tối ưu, chuyển trọng tâm sang multiplier và các lát cắt khó. |
| 2<br>(Sprint 1) | Punchline: Có model đầu tiên thắng baseline persistence, trên protocol đánh giá không leakage.<br>• Khởi động cấu phần (ii).<br>• Chốt protocol đánh giá cho toàn dự án. | • Baseline persistence và historical-average. (Hiếu)<br>• Feature engineering vòng 1: lag price, rolling mean/std, route encoding, weather, observation age. (Hiếu)<br>• Train gradient boosting cho cả hai target: price và multiplier. (Hiếu)<br>• Chốt time-based split, metric (MAE, RMSE, MAPE) và tập lát cắt báo cáo. (cả nhóm)<br>• Kiểm tra leakage bằng permutation test và shift kiểm chứng. (Chiến)<br>• Chốt KPI đề xuất từ số thực tế, gửi mentor. (cả nhóm) | • week2_report.md<br>• baseline_results.csv<br>• model_v1/<br>• eval_protocol.md<br>• leakage_check.md<br>• kpi_proposal.md | • Mọi kết quả ghi rõ config, seed, dataset version.<br>• Model v1 có bảng số so với baseline cho cả price và multiplier.<br>• Không có leakage, kiểm chứng được ghi lại.<br>• KPI proposal đã gửi mentor, phản hồi theo dõi như một gate.<br>• Gate: nếu model v1 không thắng baseline, dừng thêm feature và chẩn đoán sai số trước. |
| 3<br>(Sprint 2) | Punchline: Chốt được model tốt nhất cho (ii) và biết chính xác mô hình sai ở đâu — đầu vào cho (iii).<br>• Hoàn thiện point forecast.<br>• Trả lời: cái gì quyết định độ bất định? | • So sánh có kiểm soát các model trên cùng test set; thử ensemble nếu sai số bổ trợ nhau. (Hiếu)<br>• Ablation: bỏ weather / lịch sử giá / route để đo đóng góp từng nhóm feature. (Hiếu)<br>• Phân tích sai số theo τ và theo lát cắt: cao điểm, thời tiết xấu, surge, route hiếm. (Hiếu)<br>• EDA (iii) — Chiến: residual variance thay đổi theo feature nào, đặc biệt theo observation age.<br>• Chốt model dùng tiếp cho phần uncertainty. (cả nhóm) | • week3_report.md<br>• model_comparison.md<br>• ablation_report.md<br>• error_analysis_by_slice.md<br>• uncertainty_drivers_eda.md<br>• model_v2/ | • Bảng so sánh đầy đủ, nêu rõ model thắng và vì sao.<br>• Ablation lượng hóa đóng góp của weather, lịch sử giá và route.<br>• Chốt tối thiểu 3 biến dự báo được độ bất định.<br>• Gate: model chốt phải đạt KPI MAE; nếu không, thu hẹp scope target và ghi rõ giới hạn.<br>• Gate: nếu chưa chốt được model, cắt ensemble và sang (iii) ngay để không dồn rủi ro về cuối. |
| 4<br>(Sprint 2) | Punchline: Mô hình không còn xuất một con số mà xuất một khoảng — 80% PI đầu tiên có coverage đo được.<br>• Khởi động cấu phần (iii).<br>• So sánh sòng phẳng các hướng UQ. | • Quantile regression (pinball loss) cho các mức 10/50/90. (Chiến)<br>• Split conformal prediction trên residual của model chốt. (Chiến)<br>• Ensemble/bootstrap interval làm đối chứng. (Hiếu support)<br>• Calibration set tách biệt theo thời gian, không trùng train và test. (Chiến)<br>• Đo coverage, width, pinball loss, CRPS; vẽ reliability diagram. (Chiến) | • week4_report.md<br>• uncertainty_module/<br>• calibration_split.md<br>• uncertainty_results.csv<br>• reliability_diagrams/<br>• uq_comparison.md | • Calibration set tách biệt theo thời gian, có kiểm chứng.<br>• Cả 3 phương pháp có số coverage và width trên cùng test set.<br>• Coverage của 80% và 90% PI sai lệch ≤3 điểm phần trăm ở tối thiểu một phương pháp.<br>• Output đọc được ở dạng nghiệp vụ, ví dụ 80% PI: [17.20, 20.60].<br>• Gate: nếu không phương pháp nào đạt, ưu tiên conformal và báo cáo trung thực mức sai lệch. |
| 5<br>(Sprint 3) | Punchline: Interval nới đúng chỗ — rộng ở tình huống khó, hẹp ở tình huống dễ — và có ngưỡng cụ thể để pricing algo biết khi nào được tin tín hiệu giá đối thủ. | • Adaptive/Mondrian conformal: hiệu chuẩn theo nhóm (cao điểm, thời tiết, service, mật độ route, observation age). (Chiến)<br>• Đo conditional coverage theo từng nhóm, không chỉ marginal. (Chiến)<br>• Stress test: cao điểm, mưa/tuyết, surge cao, route ít dữ liệu, τ lớn hơn thiết kế. (Hiếu)<br>• Xây decision rule và mô phỏng tác động: tỉ lệ tín hiệu dùng được và sai số khi dùng. (cả nhóm)<br>• Đo decision latency của toàn pipeline. (Hiếu) | • week5_report.md<br>• adaptive_conformal/<br>• conditional_coverage_report.md<br>• stress_test_report.md<br>• decision_rule.md<br>• latency_report.md | • Conditional coverage báo cáo cho mọi nhóm, kèm số mẫu từng nhóm.<br>• Chứng minh bằng số rằng interval rộng hơn ở nhóm khó, hẹp hơn ở nhóm dễ.<br>• Stress test báo cáo toàn bộ kịch bản, không chọn riêng ngày tốt nhất.<br>• Decision rule có ngưỡng cụ thể và căn cứ định lượng.<br>• Gate: nếu conditional coverage hỏng ở nhóm quan trọng, hạ claim xuống mức marginal và ghi rõ điều kiện. |
| 6<br>(Sprint 3) | Punchline: Trả lời được câu hỏi vận hành — dự báo giá đối thủ tin được tới đâu, trong điều kiện nào, và khi nào pricing algo không nên dùng nó. | • Tổng hợp trade-off: độ chính xác, độ rộng interval, tỉ lệ tín hiệu dùng được. (cả nhóm)<br>• Khuyến nghị go/no-go cho việc đưa tín hiệu giá đối thủ vào pricing algo. (cả nhóm)<br>• Đề xuất tích hợp ở mức kiến trúc, kèm yêu cầu dữ liệu và tần suất cập nhật. (Hiếu)<br>• Nêu rõ giả định nào có thể vỡ khi chuyển từ data Boston sang bối cảnh Grab/Be. (Chiến)<br>• Đóng gói bàn giao để mentor chạy trên dataset GreenSM; hoàn thiện code, report, demo. (cả nhóm) | • week6_report.md<br>• final_report.md<br>• go_no_go_recommendation.md<br>• integration_proposal.md<br>• limitations_and_transfer.md<br>• handoff_package/<br>• README.md, demo và source code | • Trả lời rõ: tin được tới đâu, trong điều kiện nào, khi nào không nên dùng.<br>• Kết luận thuộc một nhóm: sẵn sàng thử trên dữ liệu GreenSM; cần thêm dữ liệu; hoặc chưa nên dùng.<br>• Gói bàn giao chạy được trên dữ liệu có schema khác, có hướng dẫn map trường.<br>• Mọi claim quan trọng có artifact hoặc evidence link.<br>• Gate: không claim hiệu quả trên thị trường thật vượt quá bằng chứng từ public dataset. |
