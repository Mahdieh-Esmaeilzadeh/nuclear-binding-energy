# پیش‌بینی انرژی بایندینگ هسته‌ای با استفاده از یادگیری عمیق

## هدف پروژه
طراحی مدل شبکه عصبی برای پیش‌بینی انرژی بایندینگ هسته‌ای بر اساس ویژگی‌های نوکلئون‌ها

## ساختار پروژه
nuclear-binding-energy

├── data/ # فایل‌های داده
|
├── notebooks/ # تحلیل داده و آزمایش مدل
|
├── scripts/ # اسکریپت‌های پردازش
|
├── models/ # مدل‌های ذخیره شده
|
├── report.pdf # گزارش نهایی
|
├── .gitignore
|
└── README.md

## دیتاست
داده‌های AMDC شامل:
- جرم اتمی (A)
- عدد اتمی (Z)
- انرژی بایندینگ (Binding Energy)

## راهنمای اجرا
```bash
# کلون ریپازیتوری
git clone https://github.com/your-username/nuclear-binding-energy.git

# نصب نیازمندی‌ها
pip install -r requirements.txt

# پیش‌پردازش داده‌ها
python scripts/data_preprocessing.py
