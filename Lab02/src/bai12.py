def get_two_state_mdp():
    return {
        0: {
            0: [(0.9, 0, 1.0, False), (0.1, 1, 0.0, False)],
            1: [(0.2, 0, 0.0, False), (0.8, 1, 2.0, False)],
        },
        1: {
            0: [(1.0, 1, -1.0, False)],
            1: [(0.7, 0, 5.0, True), (0.3, 1, 0.0, False)],
        }
    }

if __name__ == "__main__":
    mdp = get_two_state_mdp()
    print("Khởi tạo cấu trúc MDP 2 trạng thái thành công!")
