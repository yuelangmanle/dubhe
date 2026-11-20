#include <dubhe/dubhe.h>

#include <stdio.h>
#include <string.h>

int main(void) {
    char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             DUBHE_VERSION_MAJOR, DUBHE_VERSION_MINOR, DUBHE_VERSION_PATCH);
    if (strcmp(dubhe_version(), version) != 0) {
        fprintf(stderr, "C ABI 版本不一致（头文件 %s，库 %s）\n", version, dubhe_version());
        return 1;
    }

    dubhe_config config = {0};
    dubhe_engine* engine = dubhe_create(&config);
    if (!engine) {
        fprintf(stderr, "C ABI 创建引擎失败\n");
        return 1;
    }

    int32_t prompt = 0;
    int rc = dubhe_generate(engine, &prompt, 1, NULL, NULL, NULL, NULL, NULL);
    int reported_error = dubhe_last_error()[0] != '\0';
    dubhe_destroy(engine);
    if (rc != DUBHE_ERR_STATE || !reported_error) {
        fprintf(stderr, "C ABI 未加载模型时应返回状态错误，实际 %d\n", rc);
        return 1;
    }

    printf("C ABI %s 通过\n", version);
    return 0;
}
