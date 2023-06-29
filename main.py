from controller import get_scatter_origin, get_regress

if __name__ == '__main__':
    output, coef, intercept = get_regress(1)
    print(intercept)
