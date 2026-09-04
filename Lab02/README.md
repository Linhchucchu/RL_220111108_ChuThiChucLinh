# Lab02 - Markov Decision Process và Dynamic Programming

## Mục tiêu

1. Hiểu và mô phỏng được quá trình Markov Chain, kiểm tra tính hợp lệ của ma trận chuyển trạng thái.
2. Nắm vững bản chất của Reward, Return có chiết khấu và ảnh hưởng của hệ số gamma.
3. Tự cài đặt các thuật toán Dynamic Programming (DP) cơ bản:
   - Bellman Backup và Iterative Policy Evaluation.
   - Policy Improvement và Policy Iteration.
   - Value Iteration.
4. Áp dụng DP giải quyết bài toán môi trường FrozenLake-v1 (cả trường hợp is_slippery=True và False).
5. Đánh giá chất lượng của chiến lược tối ưu qua mô phỏng 1000 episode thực tế.

---

## Cấu trúc thư mục

```text
Lab02/
├── README.md
├── requirements.txt
├── src/
│   ├── bai01.py -> bai36.py
│   ├── mdp_utils.py
│   └── main.py
├── notebooks/
│   └── Lab02_220111108_ChuThiChucLinh.ipynb
├── figures/
│   ├── markov_distribution.png
│   ├── gamma_comparison.png
│   ├── value_iteration_convergence.png
│   ├── policy_iteration_convergence.png
│   └── algorithm_comparison.png
└── data/
    └── README.md
```

# Cài đặt và chạy

## cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```
## Chạy pipeline Dynamic Programming hoàn chỉnh:

```bash
python src/main.py
```
