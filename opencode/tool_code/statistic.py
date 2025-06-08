import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats import multicomp as mc

def get_header():
    header ="""import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats import multicomp as mc"""
    return header

def hypothesis_test(data=None, method=None, **kwargs):
    # 参数校验
    if method is None:
        raise ValueError("必须指定检验方法")
    
    method = method.lower()
    
    # 定义所有检验方法
    def one_sample_t_test(df, col, popmean):
        """单样本t检验：比较数据列col与总体均值popmean"""
        data = df[col].dropna()
        t_stat, p_val = stats.ttest_1samp(data, popmean)
        return {
            "test": "One-sample t-test",
            "t_stat": t_stat, 
            "p_value": p_val,
            "df": len(data)-1,
            "mean": np.mean(data),
            "popmean": popmean
        }

    def two_sample_t_test(df, col1, col2, equal_var=True):
        """双样本t检验：比较两独立样本 col1 与 col2，equal_var控制方差齐性"""
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=equal_var)
        return {
            "test": "Two-sample t-test",
            "t_stat": t_stat, 
            "p_value": p_val,
            "equal_var": equal_var,
            "n1": len(data1),
            "n2": len(data2),
            "mean1": np.mean(data1),
            "mean2": np.mean(data2)
        }

    def paired_t_test(df, col1, col2):
        """配对t检验：比较成对数据 col1 与 col2"""
        paired = df[[col1, col2]].dropna()
        t_stat, p_val = stats.ttest_rel(paired[col1], paired[col2])
        return {
            "test": "Paired t-test",
            "t_stat": t_stat, 
            "p_value": p_val,
            "n_pairs": len(paired),
            "mean_diff": np.mean(paired[col1] - paired[col2])
        }

    def anova_test(df, dependent, factor):
        """单因素方差分析（ANOVA），dependent为数值变量，factor为分组变量"""
        groups = df.groupby(factor)[dependent].apply(lambda x: x.dropna().values)
        if len(groups) < 2:
            raise ValueError("至少需要两个组进行ANOVA分析")
        f_stat, p_val = stats.f_oneway(*groups)
        
        # 计算组间信息
        group_stats = df.groupby(factor)[dependent].agg(['count', 'mean', 'std'])
        
        return {
            "test": "One-way ANOVA",
            "F_stat": f_stat, 
            "p_value": p_val,
            "n_groups": len(groups),
            "group_stats": group_stats.to_dict()
        }

    def tukey_hsd_test(df, dependent, factor, alpha=0.05):
        """Tukey HSD事后检验，用于多组两两比较"""
        df = df.copy()
        df[factor] = df[factor].astype(str)
        tukey = mc.pairwise_tukeyhsd(
            endog=df[dependent].dropna(),
            groups=df.loc[df[dependent].notna(), factor],
            alpha=alpha
        )
        return {
            "test": "Tukey HSD post-hoc test",
            "summary": tukey.summary(),
            "results": tukey
        }

    def dunnett_test(df, dependent, factor, control, alpha=0.05):
        """Dunnett检验，用于多组与对照组的比较"""
        try:
            import pingouin as pg
        except ImportError:
            raise ImportError("Dunnett检验需要安装pingouin包")
            
        df = df.copy()
        df[factor] = df[factor].astype(str)
        
        # 确保control组存在
        if control not in df[factor].unique():
            raise ValueError(f"对照组'{control}'不在分组变量中")
            
        result = pg.pairwise_tests(
            data=df,
            dv=dependent,
            between=factor,
            padjust='bonf',
            effsize='cohen',
            parametric=True,
            return_desc=True
        )
        
        # 筛选出与对照组的比较
        control_comparisons = result[
            (result['A'] == control) | (result['B'] == control)
        ]
        
        return {
            "test": "Dunnett's test",
            "control": control,
            "results": control_comparisons.to_dict('records'),
            "alpha": alpha
        }

    def mann_whitney_u_test(df, col1, col2):
        """Mann-Whitney U检验（非参数两独立样本检验）"""
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        u_stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        return {
            "test": "Mann-Whitney U test",
            "U_stat": u_stat, 
            "p_value": p_val,
            "n1": len(data1),
            "n2": len(data2),
            "median1": np.median(data1),
            "median2": np.median(data2)
        }

    def wilcoxon_test(df, col1, col2):
        """Wilcoxon符号秩检验（非参数配对样本检验）"""
        paired = df[[col1, col2]].dropna()
        stat, p_val = stats.wilcoxon(paired[col1], paired[col2])
        return {
            "test": "Wilcoxon signed-rank test",
            "statistic": stat, 
            "p_value": p_val,
            "n_pairs": len(paired),
            "median_diff": np.median(paired[col1] - paired[col2])
        }

    def kruskal_test(df, group_col, value_col):
        """Kruskal-Wallis检验（非参数多组独立样本检验）"""
        groups = df.groupby(group_col)[value_col].apply(lambda x: x.dropna().values)
        if len(groups) < 2:
            raise ValueError("至少需要两个组进行Kruskal-Wallis检验")
        h_stat, p_val = stats.kruskal(*groups)
        
        # 计算各组中位数
        group_stats = df.groupby(group_col)[value_col].agg(['count', 'median'])
        
        return {
            "test": "Kruskal-Wallis test",
            "H_stat": h_stat, 
            "p_value": p_val,
            "n_groups": len(groups),
            "group_stats": group_stats.to_dict()
        }
    
    def bonferroni_correction(pvals, alpha=0.05):
        """Bonferroni多重比较校正"""
        if not isinstance(pvals, (list, np.ndarray)):
            raise ValueError("pvals必须是一个列表或数组")
            
        reject, pvals_corrected, _, _ = multipletests(
            pvals=pvals,
            alpha=alpha,
            method='bonferroni'
        )
        
        return {
            "test": "Bonferroni correction",
            "original_pvals": pvals,
            "adjusted_pvals": pvals_corrected,
            "reject": reject,
            "alpha": alpha
        }
    
    def fdr_correction(pvals, alpha=0.05):
        """FDR（错误发现率）校正"""
        if not isinstance(pvals, (list, np.ndarray)):
            raise ValueError("pvals必须是一个列表或数组")
            
        reject, pvals_corrected, _, _ = multipletests(
            pvals=pvals,
            alpha=alpha,
            method='fdr_bh'
        )
        
        return {
            "test": "FDR correction (Benjamini-Hochberg)",
            "original_pvals": pvals,
            "adjusted_pvals": pvals_corrected,
            "reject": reject,
            "alpha": alpha
        }
    
    # 方法映射
    method_map = {
        "one_sample_t": one_sample_t_test,
        "two_sample_t": two_sample_t_test,
        "paired_t": paired_t_test,
        "anova": anova_test,
        "tukey": tukey_hsd_test,
        "dunnett": dunnett_test,
        "mann_whitney": mann_whitney_u_test,
        "wilcoxon": wilcoxon_test,
        "kruskal": kruskal_test,
        "bonferroni": bonferroni_correction,
        "fdr": fdr_correction,
    }
    
    if method not in method_map:
        raise ValueError(f"不支持的检验方法：{method}，可用方法：{list(method_map.keys())}")
    
    # 检查是否需要DataFrame
    if method not in ['bonferroni', 'fdr'] and data is None:
        raise ValueError(f"方法'{method}'需要DataFrame输入")
        
    # 执行检验
    try:
        return method_map[method](data, **kwargs) if method not in ['bonferroni', 'fdr'] else method_map[method](**kwargs)
    except Exception as e:
        raise ValueError(f"执行检验时出错: {str(e)}")
    

def distribution_test(data, col, method="shapiro", **kwargs):
    # 提取数据，并去除缺失值
    data = data[col].dropna().values
    
    if method.lower() == "shapiro":
        stat, p_value = stats.shapiro(data)
        return {"statistic": stat, "p_value": p_value}
    
    elif method.lower() == "ks":
        # 要求传入 'dist' 参数指定理论分布名称，如 "norm"
        dist_name = kwargs.get("dist", None)
        if dist_name is None:
            raise ValueError("使用Kolmogorov-Smirnov检验时，请在kwargs中指定 'dist' 参数，如 dist='norm'")
        # 可传入理论分布的参数，如loc, scale或args，默认取自kwargs
        # 如果指定了args，则使用args，否则尝试读取loc和scale
        args = kwargs.get("args", ())
        loc = kwargs.get("loc", 0)
        scale = kwargs.get("scale", 1)
        # 构造理论分布对象
        try:
            dist = getattr(stats, dist_name)
        except AttributeError:
            raise ValueError(f"scipy.stats中不存在分布：{dist_name}")
        # 进行KS检验：注意，ks检验需要提供累积分布函数（cdf）
        stat, p_value = stats.kstest(data, dist_name, args=args, alternative=kwargs.get("alternative", 'two-sided'))
        return {"statistic": stat, "p_value": p_value}
    
    else:
        raise ValueError(f"不支持的检验方法：{method}")