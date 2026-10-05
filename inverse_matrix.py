"""역행렬 계산 프로그램: 행렬식(수반행렬) 방식 vs 가우스-조던 소거법 비교"""
from fractions import Fraction
import time


# ---------- 1. 행렬 입력 ----------
def input_matrix():
    n = int(input("행렬의 크기 n을 입력하세요: "))
    if n < 1:
        raise ValueError("n은 1 이상의 정수여야 합니다.")
    print(f"{n} x {n} 행렬을 행 단위로 입력하세요 (공백으로 구분).")
    matrix = []
    for i in range(n):
        row = list(map(int, input(f"{i + 1}행: ").split()))
        if len(row) != n:
            raise ValueError(f"각 행에는 정확히 {n}개의 정수가 필요합니다.")
        matrix.append(row)
    return matrix


# ---------- 2. 행렬식을 이용한 역행렬 ----------
def minor(m, i, j):
    return [r[:j] + r[j + 1:] for k, r in enumerate(m) if k != i]


def determinant(m):
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return sum((-1) ** j * m[0][j] * determinant(minor(m, 0, j)) for j in range(n))


def inverse_by_determinant(m):
    n = len(m)
    det = determinant(m)
    if det == 0:
        raise ValueError("행렬식이 0이므로 역행렬이 존재하지 않습니다.")
    if n == 1:
        return [[Fraction(1, det)]]
    # 수반행렬 = 여인수행렬의 전치
    return [[Fraction((-1) ** (i + j) * determinant(minor(m, j, i)), det)
             for j in range(n)] for i in range(n)]


# ---------- 3. 가우스-조던 소거법 ----------
def inverse_by_gauss_jordan(m):
    n = len(m)
    # 확대행렬 [A | I]
    aug = [[Fraction(m[i][j]) for j in range(n)] +
           [Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("피벗이 없으므로 역행렬이 존재하지 않습니다.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


# ---------- 4. 출력 및 비교 ----------
def fmt(x):
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def print_matrix(title, m, as_decimal=False):
    print(title)
    cells = [[f"{float(x):.4f}" if as_decimal else fmt(x) for x in r] for r in m]
    w = max(len(c) for r in cells for c in r)
    for r in cells:
        print("  [ " + "  ".join(c.rjust(w) for c in r) + " ]")


def multiply(a, b):
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def is_identity(m):
    return all(m[i][j] == (1 if i == j else 0) for i in range(len(m)) for j in range(len(m)))


def main():
    A = input_matrix()
    print_matrix("\n[입력 행렬 A]", [[Fraction(x) for x in r] for r in A])

    results = {}
    for name, func in (("행렬식 방법", inverse_by_determinant),
                       ("가우스-조던 방법", inverse_by_gauss_jordan)):
        try:
            t0 = time.perf_counter()
            results[name] = func(A)
            elapsed = (time.perf_counter() - t0) * 1000
            print_matrix(f"\n[{name}] 역행렬 (분수)", results[name])
            print_matrix(f"[{name}] 역행렬 (소수)", results[name], as_decimal=True)
            print(f"  계산 시간: {elapsed:.4f} ms")
        except ValueError as e:
            print(f"\n[{name}] 오류: {e}")

    if len(results) == 2:
        d, g = results["행렬식 방법"], results["가우스-조던 방법"]
        print("\n[결과 비교]", "두 방법의 결과가 동일합니다." if d == g
              else "두 방법의 결과가 다릅니다.")
        # 추가 기능: 검증 A x A^-1 = I
        ok = is_identity(multiply([[Fraction(x) for x in r] for r in A], g))
        print("[검증] A x A^-1 = I :", "성립 (정상)" if ok else "불성립")
    else:
        print("\n[결과 비교] 역행렬이 존재하지 않아 비교할 수 없습니다.")


if __name__ == "__main__":
    main()
