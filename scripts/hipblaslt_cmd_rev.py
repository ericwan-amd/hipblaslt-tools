import pandas as pd
from io import BytesIO

# Define the CSV data
csv_data = """transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,aux_type,hipblaslt-Gflops,hipblaslt-GB/s,us,CPU-Gflops,CPU-us,norm_error,atol,rtol
T,N,0,1,5120,9951,8192,1,8192,41943040,0,8192,81518592,5120,50949120,5120,50949120,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,0,0,none,0,f16_r,f16_r,0,0,0,75246.1,29.284,11093.6
T,N,0,1,5120,9951,7168,1,7168,36700160,0,7168,71328768,5120,50949120,5120,50949120,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,0,0,none,0,f16_r,f16_r,0,0,0,136026,55.1474,5369.6
T,N,0,1,5120,9951,9216,1,9216,47185920,0,9216,91708416,5120,50949120,5120,50949120,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,0,0,none,0,f16_r,f16_r,0,0,0,136894,51.5468,6860
"""

# Convert to DataFrame
df = pd.read_csv(BytesIO(csv_data.encode()))

# Save to Excel
output_file = "gfx1201_tn_tuning.xlsx"
df.to_excel(output_file, index=False, sheet_name="gfx1201_TN_Tuning")

print(f"Excel file saved as: {output_file}")