import difflib

def code_similarity(code1: str, code2: str) -> float:
    matcher = difflib.SequenceMatcher(None, code1, code2)
    return matcher.ratio()

if __name__ == "__main__":
    ground_truth = """
    #include <stdlib.h>
    #include <stdio.h>
    #include <omp.h> 
    int dummyMethod1();
    int dummyMethod2();
    int dummyMethod3();
    int dummyMethod4();

    int main(int argc,char *argv[])
    {
    int i;
    int len = 100;
    int a[100];
    dummyMethod3();
    
    #pragma omp parallel for private (i)
    //#pragma rose_outline
    for (i = 0; i <= len - 1; i += 1) {
        a[i] = i;
    }
    dummyMethod4();
    dummyMethod1();
    for (i = 0; i <= len - 1 - 1; i += 1) {
        a[i + 1] = a[i] + 1;
    }
    dummyMethod2();
    printf("a[50]=%d\n",a[50]);
    return 0;
    }

    int dummyMethod1()
    {
    return 0;
    }

    int dummyMethod2()
    {
    return 0;
    }

    int dummyMethod3()
    {
    return 0;
    }

    int dummyMethod4()
    {
    return 0;
    }
    """
    
    
    translation = """
    #include <stdlib.h>
    #include <stdio.h>
    #include <omp.h>  // OpenMP header

    int dummyMethod1();
    int dummyMethod2();
    int dummyMethod3();
    int dummyMethod4();

    int main(int argc, char *argv[])
    {
        int i;
        int len = 100;
        int a[100];
        dummyMethod3();

        // Parallelize the initialization loop.
        #pragma omp parallel for
        for (i = 0; i < len; i++) {
            a[i] = i;
        }
        dummyMethod4();
        dummyMethod1();

        // This loop has a dependency: each iteration uses a[i] computed by a previous iteration.
        // Therefore, we keep it sequential.
        for (i = 0; i < len - 1; i++) {
            a[i + 1] = a[i] + 1;
        }
        dummyMethod2();
        printf("a[50]=%d\n", a[50]);
        return 0;
    }

    int dummyMethod1() { return 0; }
    int dummyMethod2() { return 0; }
    int dummyMethod3() { return 0; }
    int dummyMethod4() { return 0; }
    """


    print(code_similarity(ground_truth, translation))
