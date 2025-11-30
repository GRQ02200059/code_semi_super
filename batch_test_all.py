#!/usr/bin/env python
"""批量测试所有实验的最佳checkpoint"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def find_best_checkpoint(exp_dir):
    """找到最佳IoU checkpoint"""
    iou_dir = exp_dir / "version_0" / "weights" / "iou"
    if not iou_dir.exists():
        return None
    
    checkpoints = list(iou_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    
    # 按IoU值排序，取最高的
    best_ckpt = None
    best_iou = 0.0
    for ckpt in checkpoints:
        # 解析文件名中的IoU值
        name = ckpt.name
        if "val_iou=" in name:
            try:
                iou_str = name.split("val_iou=")[1].replace(".ckpt", "")
                iou = float(iou_str)
                if iou > best_iou:
                    best_iou = iou
                    best_ckpt = ckpt
            except:
                pass
    
    if best_ckpt is None:
        return None
    return best_ckpt, best_iou

def get_experiment_info(exp_dir):
    """获取实验信息"""
    config_path = exp_dir / "version_0" / "config.py"
    info = {"name": exp_dir.name}
    
    if config_path.exists():
        try:
            content = config_path.read_text()
            # 简单解析一些关键信息
            if "use_contrastive" in content and "True" in content:
                info["contrastive"] = True
            if "use_uncertainty" in content and "True" in content:
                info["uncertainty"] = True
            if "use_multiscale" in content and "True" in content:
                info["multiscale"] = True
        except:
            pass
    
    return info

def main():
    outputs_dir = Path("outputs")
    results = []
    
    # 找到所有10月和11月的实验
    experiments = []
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        # 检查是否是10月或11月的实验
        if "202510" in name or "202511" in name:
            ckpt_result = find_best_checkpoint(exp_dir)
            if ckpt_result:
                best_ckpt, best_val_iou = ckpt_result
                experiments.append({
                    "dir": exp_dir,
                    "ckpt": best_ckpt,
                    "val_iou": best_val_iou
                })
    
    print(f"\n{'='*80}")
    print(f"找到 {len(experiments)} 个有效实验")
    print(f"{'='*80}\n")
    
    # 测试每个实验
    for i, exp in enumerate(experiments):
        exp_dir = exp["dir"]
        ckpt = exp["ckpt"]
        val_iou = exp["val_iou"]
        
        print(f"\n[{i+1}/{len(experiments)}] 测试: {exp_dir.name}")
        print(f"  Checkpoint: {ckpt.name}")
        print(f"  Val IoU: {val_iou:.4f}")
        
        # 运行测试
        cmd = [
            "python", "tools/launch.py", "test",
            "-e", str(exp_dir / "version_0"),
            "--ckpt-path", str(ckpt)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 解析输出中的测试结果
            output = result.stdout + result.stderr
            test_iou = None
            test_f1 = None
            
            for line in output.split("\n"):
                if "test_iou" in line.lower():
                    try:
                        # 尝试解析IoU值
                        parts = line.split()
                        for j, p in enumerate(parts):
                            if "iou" in p.lower() and j+1 < len(parts):
                                test_iou = float(parts[j+1].strip(",:"))
                                break
                    except:
                        pass
                if "test_f1" in line.lower():
                    try:
                        parts = line.split()
                        for j, p in enumerate(parts):
                            if "f1" in p.lower() and j+1 < len(parts):
                                test_f1 = float(parts[j+1].strip(",:"))
                                break
                    except:
                        pass
            
            results.append({
                "name": exp_dir.name,
                "val_iou": val_iou,
                "test_iou": test_iou,
                "test_f1": test_f1,
                "status": "success" if result.returncode == 0 else "failed",
                "checkpoint": ckpt.name
            })
            
            if test_iou:
                print(f"  ✅ Test IoU: {test_iou:.4f}")
            else:
                print(f"  ⚠️ 测试完成，但无法解析IoU")
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ 超时")
            results.append({
                "name": exp_dir.name,
                "val_iou": val_iou,
                "status": "timeout",
                "checkpoint": ckpt.name
            })
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results.append({
                "name": exp_dir.name,
                "val_iou": val_iou,
                "status": f"error: {e}",
                "checkpoint": ckpt.name
            })
    
    # 输出汇总
    print(f"\n{'='*80}")
    print("测试结果汇总")
    print(f"{'='*80}")
    print(f"{'实验名称':<60} {'Val IoU':>10} {'Test IoU':>10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x.get("test_iou") or 0, reverse=True):
        val_iou = f"{r['val_iou']:.4f}" if r.get('val_iou') else "N/A"
        test_iou = f"{r['test_iou']:.4f}" if r.get('test_iou') else "N/A"
        print(f"{r['name']:<60} {val_iou:>10} {test_iou:>10}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"test_results_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
