for data in wot root moon
do 
    for method in TrajectoryNet FBSDE
    do
        sbatch --job-name="workflow-$data-$method" workflow.sh $data $method
    done
done