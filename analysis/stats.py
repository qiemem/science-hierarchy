from bootstrap import bootstrap
from pylab import *
import csv
from scipy import stats

BOOTSTRAP_SIZE = 10000

print('Analyzing metric validation')
with open('exteuc_validation/results.csv') as f:
    metric_sample = np.array([int(q['response'])-1 for q in csv.DictReader(f)])
print('Sample size: {}'.format(len(metric_sample)))
print('Mean: {}'.format(metric_sample.mean()))
print('P-value (Binomial test): {}'.format(stats.binom_test(np.count_nonzero(metric_sample), len(metric_sample))))
#metric_means = bootstrap(metric_sample, BOOTSTRAP_SIZE)
#print('Bootstrap mean: {}, Standard deviation: {}'.format(metric_means.mean(), metric_means.std()))
#print('T-Statistic: {}, P-Value: {}'.format(*stats.ttest_1samp(metric_means, .5)))
#metricfig = figure()
#hist(metric_means, bins=20)
#title('Metric Validation Bootstrapped Histogram')
#xlabel('% agreement with metric')
#ylabel('Count')
#metricfig.savefig('metric_bootstrap.png')

print('Analyzing cluster validation')
with open('cluster_validation/results.csv') as f:
    cluster_sample = np.array([int(q['response'])-1 for q in csv.DictReader(f)])
print('Sample size: {}'.format(len(cluster_sample)))
print('Mean: {}'.format(cluster_sample.mean()))
print('P-value (Binomial test): {}'.format(stats.binom_test(np.count_nonzero(cluster_sample), len(cluster_sample))))
#cluster_means = bootstrap(cluster_sample, BOOTSTRAP_SIZE)
#print('Bootstrap mean: {}, Standard deviation: {}'.format(cluster_means.mean(), cluster_means.std()))
#print('T-Statistic: {}, P-Value: {}'.format(*stats.ttest_1samp(cluster_means, .5)))
#clusterfig = figure()
#hist(cluster_means, bins=20)
#title('Cluster Validation Bootstrapped Histogram')
#xlabel('% agreement with clustering')
#ylabel('Count')
#clusterfig.savefig('cluster_bootstrap.png')
