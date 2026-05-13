#include <stdio.h>

int main() {
    int a, b, sum;
    printf("请输入两个整数，用空格分隔: ");
    scanf("%d %d", &a, &b);
    sum = a + b;
    printf("它们的和是: %d\n", sum);
    return 0;
}