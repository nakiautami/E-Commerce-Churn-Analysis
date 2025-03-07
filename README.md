# E-Commerce-Churn-Analysis
Oleh:
-  Azhar Maulana
- Eki Nakia Utami
- Naila Firdusi Putri Fadilah

**INTRODUCTION**

Project ini bertujuan untuk memprediksi churn pada ecommerce menggunakan machine learning. Di dalam project ini kita akan mencari cara cleaning terbaik, model terbaik, dan parameter terbaik agar menghasilkan prediksi yang terbaik yang didasarkan dari fitur-fitur telah disediakan dalam dataset. 
Daftar Isi 
1.	Business Problem Understanding
2.	Data Understanding
3.	Data Cleaning
4.	Data Analisis
5.	Data Pre-Processing
6.	Modelling
7.	Evaluation
8.	Conclusion
9.	Recommendation

**BUSINESS PROBLEM UNDERSTANDING**
1.	Context
- E-Commerce atau perdagangan elektronik adalah proses pembelian dan penjualan barang atau jasa melalui media elektronik, khususnya internet. Dalam e-commerce, transaksi dilakukan secara online tanpa interaksi fisik antara penjual dan pembeli. Keberadaan platform digital mempermudah perusahaan dalam menjangkau pasar yang lebih luas dengan biaya yang lebih efisien. Namun, dengan pesatnya pertumbuhan e-commerce, persaingan semakin ketat, sehingga perusahaan menghadapi tantangan dalam meningkatkan serta mempertahankan pelanggan.

2. Problem Statement
- Banyak perusahaan e-commerce menghadapi kesulitan dalam mengelola churn pelanggan karena kurangnya pendekatan berbasis data dalam strategi retensi. Tanpa analisis yang tepat, **perusahaan sering kali mengalokasikan sumber daya secara tidak efisien untuk mempertahankan pelanggan yang berisiko churn**. Selain itu, tidak adanya model prediksi churn yang akurat dapat menyebabkan keputusan bisnis yang kurang efektif dalam upaya mempertahankan pelanggan. **Oleh karena itu, diperlukan solusi berbasis machine learning untuk mengidentifikasi pelanggan yang berisiko churn dan mengembangkan strategi retensi yang lebih tepat sasaran.**

3. Rumusan Masalah
-	Bagaimana perilaku pelanggan yang berisiko churn dapat dianalisis secara efektif?
- Faktor-faktor apa saja yang paling berpengaruh dalam menentukan apakah pelanggan akan churn?
- Bagaimana model machine learning dapat digunakan untuk memprediksi kemungkinan pelanggan akan churn?
- Bagaimana strategi retensi yang efektif berdasarkan hasil prediksi churn?

4. Goals
- Dengan menganalisis perilaku pelanggan serta memprediksi kemungkinan churn menggunakan algoritma machine learning, diharapkan perusahaan dapat mengembangkan model prediksi churn yang lebih akurat. Model ini diharapkan dapat membantu manajemen dalam meningkatkan efektivitas strategi retensi pelanggan melalui personalisasi penawaran dan efisiensi operasional.
Selain strategi retensi, model yang dikembangkan juga dapat mendukung pengambilan keputusan dalam segmentasi pelanggan, perancangan program loyalitas, serta pengelolaan profitabilitas pendapatan. Dengan mengklasifikasikan pelanggan berdasarkan kemungkinan churn, perusahaan dapat secara proaktif menangani potensi churn, misalnya dengan menawarkan insentif khusus atau meningkatkan kualitas layanan bagi pelanggan berisiko tinggi

5. Analytic Approach
- Kita akan menganalisis data untuk menemukan pola yang membedakan pelanggan yang churn dan tidak. Setelah itu, kita akan membangun model klasifikasi untuk memprediksi kemungkinan pelanggan akan churn, sehingga perusahaan dapat mengambil langkah yang tepat.

6. Target
- Dalam analisis churn, target diklasifikasikan sebagai berikut:
  - 0: Customer tidak churn → Pelanggan masih aktif menggunakan layanan atau produk dalam periode tertentu.
  - 1: Customer churn → Pelanggan berhenti menggunakan layanan, ditandai dengan: 
      - Tidak ada transaksi dalam jangka waktu tertentu.
      - Berhenti berlangganan.
      - Tidak lagi menggunakan aplikasi/platform.
      -	Tidak merespons kampanye pemasaran.

7. Metric Evaluation
- Biaya Churn
   - Customer Acquisition Cost (CAC) → Biaya memperoleh pelanggan baru
   - Customer Retention Cost (CRC) → Biaya mempertahankan pelanggan lama
     
- Kesalahan Prediksi Churn
  - Confusion matrix mengklasifikasikan prediksi churn menjadi:
    - False Positive (FP) → Type 1 Error: Pelanggan tetap diprediksi churn → meningkatkan biaya retensi.
    - False Negative (FN) → Type 2 Error: Pelanggan churn diprediksi tetap → kehilangan pelanggan & meningkatkan CAC.
  - Dalam konteks analisis churn, **Type 2 Error lebih krusial untuk dihindari karena dampaknya yang lebih besar terhadap profitabilitas perusahaan**. Ketidakmampuan mengidentifikasi pelanggan yang berisiko churn dapat menyebabkan hilangnya pelanggan dan menignkatkan CAC. Sementara itu, meskipun **Type 1 Error juga perlu diperhatikan, efeknya cenderung lebih terkait pemborosan sumber daya tetapi tidak langsung mengurangi jumlah pelanggan**.

- Perbandingan CAC vs CRC
  - CAC lebih tinggi: Contoh, jika perusahaan menghabiskan $90 per pelanggan untuk akuisisi.
  - CRC lebih rendah: Contoh, hanya $40 per pelanggan untuk retensi.
  - Strategi retensi lebih efisien dalam jangka panjang dibandingkan akuisisi pelanggan baru.

- Optimasi Churn dengan F2-Score
  - F2-Score menekankan recall, sehingga lebih efektif dalam mengidentifikasi pelanggan yang berisiko churn.
  - Rumus F2-Score:
    - $$F_2= \frac{(1+2^2) \times \text{Precision} \times \text{Recall}}{2^2 \times \text{Precision} \times \text{Recall}}$$
  - F2-Score membantu mengurangi churn dengan menyeimbangkan kesalahan prediksi dan mengoptimalkan strategi retensi.
  - Accuracy tinggi → Model mampu memprediksi churn dan non-churn dengan baik.

  **Data Understanding**
  - Data Set
    - Data Source : https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
      - Dataset tidak seimbang
      - Setiap baris data mempresentasikan informasi customer E-commerce
   - Secara umum kita bisa melihat data:
     - Dataset ini memiliki 5.630 baris(entri) dan 20 kolom.
     - `PreferredPaymentMode`: Terdapat metode pembayaran seperti Debit Card, UPI, dan CC (Credit Card). 
     - `CityTier`: Istilah ini sering digunakan dalam segmentasi kota di Amerika Serikat, di mana kota dibagi menjadi Tier 1, Tier 2, dan Tier 3.

    Atribut Information

    | Attribute | Data Type, Length | Description |
    | --- | --- | --- |
    | CustomerID | Integer | ID unik pelanggan|
    | Churn | Integer | status apakah pelanggan churn atau tidak  |
    | Tenure | Float | Durasi pelanggan dalam layanan |
    | PreferredLoginDevice | Object | perangkat yang digunakan ketika login |
    | CityTier | Integer | Tingkat kota customer|
    | WarehouseToHome | Float | Jarak antara gudang penyimpanan dan rumah pelanggan |
    | PreferredPaymentMode | Object | Metode pembayaran yang digunakan pelanggan |
    | Gender | Object | Jenis kelamin pelanggan  |
    | HourSpendOnApp | Float | lama waktu customer menggunakan aplikasi |
    | NumberOfDeviceRegistered | Integer | Jumlah perangkat yang terdaftar pada setiap customer|
    | PreferedOrderCat | Object | Kategori produk yang sering dipesan customer satu bulan terakhir |
    | SatisfactionScore | Integer | Skor kepuasan pelanggan terhadap layanan|
    | MaritalStatus | Object | Status pernikahan pelanggan|
    | NumberOfAddress | Integer | Jumlah alamat yang didaftarkan customer |
    | Complain | Integer | Apakah pelanggan mengajukan keluhan dalam satu bulan terakhir|
    | OrderAmountHikeFromlastYear | Float | Persentase peningkatan jumlah pesanan dibandingkan tahun lalu  |
    | CouponUsed | Float | Jumlah kupon/voucher yang telah digunakan customer selama satu bulan terakhir |
    | OrderCount | Float | Jumlah pesanan yang dilakukan selama satu bulan terakhir |
    | DaySinceLastOrder | Float | Jumlah hari sejak pesanan terakhir pelanggan|
    | CashbackAmount | Float | Rata-rata cashback yang diterima dalam satu bulan terakhir |

  **DATA CLEANING**
      Berikut ini hasil pada tahapan data cleaning.
    | Feature | Missing Value | Outlier | Inkonsistensi Data|
    | --- | --- | --- | --- |
    |`Tenure` | Median | Drop value > 31| - |
    | `WarehouseToHome` | Median | Drop value > 40 | - |
    | `HourSpendOnApp` | Median | Outlier di pertahankan | - |
    | `NumberOfDeviceRegistered` | - | Outlier di pertahankan | - |
    | `NumberOfAddress` | - | Outlier di pertahankan | - |
    | `OrderAmountHikeFromlastYear` | Median | Outlier di pertahankan | Mengubah nama kolom 'OrderAmountHikeFromlastYear' menjadi       `OrderAmountHikeFromLastYear `|
    | `CouponUsed` | Median berdasarkan OrderCount | Outlier di pertahankan | - |
    | `OrderCount` | Median berdasarkan CouponUsed | Outlier di pertahankan | - |
    | `DaySinceLastOrder` | Median berdasarkan OrderCount | Drop value > 31 | - |
    | `CashbackAmount` | - | Outlier di pertahankan | - |
    | `PreferredLoginDevice` | - | - | nilai "Phone" akan disesuaikan menjadi "Mobile Phone |  
    | `PreferredPaymentMode` | - | - | Mengubah nilai "COD" menjadi "Cash on Delivery dan nilai "CC" perlu disamakan menjadi "Credit Card  |
    | `PreferredOrderCat` |  |  | Mengubah nama kolom 'PreferedOrderCat' menjadi `PreferredOrderCat ` |
