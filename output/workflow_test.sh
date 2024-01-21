for data in wot root moon
do
    for i_kl in 0 1 2
    do
        for i_lock in 0 1 2
        do
            for i_ent in 0 1 2
            do
                for i_s in 0 1 2
                do
                    sbatch --job-name="single-$data-FBSDE_minus-$i_kl$i_lock$i_ent$i_s" job_single.sh $data FBSDE_minus $i_kl $i_lock $i_ent $i_s
                done
            done
        done
    done
done