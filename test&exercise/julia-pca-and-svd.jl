"""
线性组合在数据科学中有重要作用,
当我们看到一个矩阵, 我们想要知道通过那些向量的线性组合可以近似的生成这个矩阵. 
最好是分解出来像收音机的调谐旋钮一样, 有些负责大幅度变化, 有些适应小幅度变化. 
收音机的波段也是一个映射关系.
"""

using LinearAlgebra,Latexify

A = [0.0  1.0 -2.3  0.1;
     1.3  4.0 -0.1  0.0;
     4.1 -1.0  0.0 1.7]
    
"""
    svd_block(A::Matrix)

```julia
function svd_block(A::Matrix)
    U,Σ,V=svd(A)
    return (i)->U[:,i]*Σ[i]*V[:,i]'
end
```
svd_block 是高阶函数 ,接收一个矩阵, 返回一个函数,接收索引然后返回
主成分的 block 矩阵,主要的 block相加可以得到原始矩阵的近似低维度表示

TBW
"""
function svd_block(A::Matrix)
        U,Σ,V=svd(A)
		return (i)->U[:,i]*Σ[i]*V[:,i]'
end
	
block=svd_block(A)
b1=block(1)
b2=block(2)
b3=block(3)
(b1+b2).|>(d->round(d,digits=2))
