for n_t in 'outlinear'
    do
        for N in 2000 4000 6000 8000
            do
                python -u /home/aistudio/work/ns/unsteady_NS_std.py --noise_type $n_t --noise 0 --abnormal_ratio 0.05 --N $N --weight 1E-0 --save_path "./data/size/""$n_t" | tee $"data/log/""$n_t_""$N"".log"    
            done     
    done

for n_t in 't1' 'contamined' 'normal' 'none'
    do
        for N in 2000 4000 6000 8000
            do
                if [ $n_t = 'none' ] && [ $N = 2000 ]
                then
                    echo "skip none 2000"
                else    
                    python -u /home/aistudio/work/ns/unsteady_NS_std.py --noise_type $n_t --noise 0.05 --abnormal_ratio 0 --N $N --weight 1E-0 --save_path "./data/size/""$n_t" | tee $"data/log/""$n_t_""$N"".log"
                fi     
            done   
    done

for n_t in 'outlinear'
    do
        for N in 2000 4000 6000 8000
            do      
                python -u /home/aistudio/work/ns/unsteady_NS_std.py --noise_type $n_t --noise 0.05 --abnormal_ratio 0.05 --N $N --weight 1E-0 --save_path "./data/size/""$n_t" | tee $"data/log/""$n_t_""$N"".log" 
            done
        
    done
