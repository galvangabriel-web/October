# Post-Race Analysis - Maintenance Guide

**Purpose:** Ongoing maintenance and monitoring of the Post-Race Analysis fix
**Audience:** System administrators, DevOps, maintainers
**Update Frequency:** Monthly or after significant changes

---

## Table of Contents

1. [Daily Monitoring](#daily-monitoring)
2. [Weekly Health Checks](#weekly-health-checks)
3. [Monthly Maintenance](#monthly-maintenance)
4. [Update Procedures](#update-procedures)
5. [Monitoring Metrics](#monitoring-metrics)
6. [Alert Thresholds](#alert-thresholds)

---

## Daily Monitoring

### Automated Health Check Script

**File:** `health_check_post_race.sh`

```bash
#!/bin/bash
# Daily health check for Post-Race Analysis tab
# Run via cron: 0 9 * * * /path/to/health_check_post_race.sh

echo "=== Post-Race Analysis Health Check ===" > /var/log/post_race_health.log
echo "Date: $(date)" >> /var/log/post_race_health.log
echo "" >> /var/log/post_race_health.log

# 1. Dashboard Status
echo "[1] Dashboard Status:" >> /var/log/post_race_health.log
curl -s -o /dev/null -w "%{http_code}" http://200.58.107.214:8050 >> /var/log/post_race_health.log
echo " (200 = OK)" >> /var/log/post_race_health.log

# 2. Process Check
echo "" >> /var/log/post_race_health.log
echo "[2] Process Count:" >> /var/log/post_race_health.log
ssh tactical@200.58.107.214 -p 5197 "pgrep -f dashboard | wc -l" >> /var/log/post_race_health.log

# 3. Error Count (last 24h)
echo "" >> /var/log/post_race_health.log
echo "[3] Errors (last 24h):" >> /var/log/post_race_health.log
ssh tactical@200.58.107.214 -p 5197 "grep -c 'ERROR\|Exception' /home/tactical/racing_analytics/dashboard.log 2>/dev/null || echo 0" >> /var/log/post_race_health.log

# 4. Predictor Check
echo "" >> /var/log/post_race_health.log
echo "[4] Active Predictor:" >> /var/log/post_race_health.log
ssh tactical@200.58.107.214 -p 5197 "grep 'Using.*Predictor' /home/tactical/racing_analytics/dashboard.log | tail -1" >> /var/log/post_race_health.log

# 5. Alert if issues
ERROR_COUNT=$(ssh tactical@200.58.107.214 -p 5197 "grep -c 'ERROR\|Exception' /home/tactical/racing_analytics/dashboard.log 2>/dev/null || echo 0")
if [ "$ERROR_COUNT" -gt 10 ]; then
    echo "" >> /var/log/post_race_health.log
    echo "⚠️  ALERT: High error count ($ERROR_COUNT errors in last 24h)" >> /var/log/post_race_health.log
    # Send email or notification here
fi

echo "" >> /var/log/post_race_health.log
echo "=== End Health Check ===" >> /var/log/post_race_health.log

# Show results
cat /var/log/post_race_health.log
```

### Manual Daily Check (5 minutes)

```bash
# 1. Check dashboard is up
curl http://200.58.107.214:8050

# 2. Check recent errors
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "grep -E 'ERROR|Exception' /home/tactical/racing_analytics/dashboard.log | tail -20"

# 3. If errors found, investigate
# See TROUBLESHOOTING_GUIDE.md
```

---

## Weekly Health Checks

### Every Monday Morning

#### 1. Verify Fix is Still Active

```bash
cd data_analisys_car

# Check SimplePostRacePredictor exists
venv/Scripts/python.exe ssh_helper.py "ls -la /home/tactical/racing_analytics/src/models/inference/simple_post_race_predictor.py"

# Check it's being used
venv/Scripts/python.exe ssh_helper.py "grep 'SimplePostRacePredictor' /home/tactical/racing_analytics/dashboard.log | tail -5"
```

**Expected:**
- ✅ File exists (8.4 KB)
- ✅ Logs show "Using SimplePostRacePredictor"

#### 2. Test with Template CSV

```bash
# Upload post_race_sample_template.csv to dashboard
# Navigate to: http://200.58.107.214:8050
# Tab: Post-Race Analysis
# Click: Analyze Session

# Expected:
# ✅ Analysis completes
# ✅ Charts display
# ✅ No error messages
```

#### 3. Check Disk Space

```bash
cd data_analisys_car
venv/Scripts/python.exe ssh_helper.py "df -h /home/tactical/racing_analytics"

# Alert if <10% free space
```

#### 4. Review Performance Metrics

```bash
# Average prediction time (should be <1 second)
venv/Scripts/python.exe ssh_helper.py "grep 'Predictions generated' /home/tactical/racing_analytics/dashboard.log | tail -20"

# Check for slowdowns
```

---

## Monthly Maintenance

### First Monday of Each Month

#### 1. Log Rotation

```bash
cd data_analisys_car

# Archive old logs
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && mv dashboard.log dashboard.log.$(date +%Y%m).bak && touch dashboard.log"

# Compress old logs
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && gzip dashboard.log.*.bak"

# Delete logs older than 6 months
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && find . -name 'dashboard.log.*.bak.gz' -mtime +180 -delete"
```

#### 2. Performance Review

```bash
# Count total predictions this month
venv/Scripts/python.exe ssh_helper.py "grep -c 'Predictions generated' /home/tactical/racing_analytics/dashboard.log"

# Count errors this month
venv/Scripts/python.exe ssh_helper.py "grep -c 'ERROR' /home/tactical/racing_analytics/dashboard.log"

# Calculate error rate
# Error rate should be <1%
```

#### 3. Model File Verification

```bash
# Check model file hasn't been corrupted
venv/Scripts/python.exe ssh_helper.py "md5sum /home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl"

# Compare with known good hash
# Expected: (record hash on first deployment)
```

#### 4. Dependency Updates

```bash
# Check for security updates
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && venv/bin/pip list --outdated"

# Update critical security patches only
# Test in staging first!
```

---

## Update Procedures

### Updating SimplePostRacePredictor

**When to Update:**
- Bug fixes identified
- Performance improvements available
- New features needed

**Procedure:**

```bash
# 1. Make changes locally
cd data_analisys_car/src/models/inference
# Edit simple_post_race_predictor.py

# 2. Test locally
cd ../..
venv/Scripts/python.exe test_simple_predictor.py
# Must pass before deploying

# 3. Backup production
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && cp src/models/inference/simple_post_race_predictor.py src/models/inference/simple_post_race_predictor.py.backup.$(date +%Y%m%d)"

# 4. Deploy
venv/Scripts/python.exe deploy_post_race_fix.py

# 5. Monitor logs for 15 minutes
venv/Scripts/python.exe ssh_helper.py "tail -f /home/tactical/racing_analytics/dashboard.log"
# Watch for errors

# 6. Test with template CSV
# Upload post_race_sample_template.csv
# Verify success

# 7. If issues, rollback
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && mv src/models/inference/simple_post_race_predictor.py.backup.YYYYMMDD src/models/inference/simple_post_race_predictor.py"
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

### Updating Model File

**When to Update:**
- Model retrained with new data
- Improved model available

**Procedure:**

```bash
# 1. Test new model locally
cd data_analisys_car
# Place new model in data/models/lightgbm_baseline_new.pkl

# Modify test script to use new model
# Run: venv/Scripts/python.exe test_simple_predictor.py

# 2. Compare performance
# Old model: 95-96% R²
# New model: Should be >= 95% R²

# 3. Backup old model on server
venv/Scripts/python.exe ssh_helper.py "cp /home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl /home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl.backup.$(date +%Y%m%d)"

# 4. Upload new model
cd data_analisys_car
venv/Scripts/python.exe -c "
import paramiko, os
from dotenv import load_dotenv
load_dotenv()
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('200.58.107.214', 5197, 'tactical', os.getenv('SSH_PASSWORD'))
sftp = ssh.open_sftp()
sftp.put('data/models/lightgbm_baseline_new.pkl', '/home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl')
sftp.close()
ssh.close()
"

# 5. Restart dashboard
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"

# 6. Test thoroughly
# Upload multiple CSVs, verify predictions
```

---

## Monitoring Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Alert Threshold | Check Frequency |
|--------|--------|-----------------|-----------------|
| Dashboard Uptime | 99.9% | <95% | Daily |
| Prediction Success Rate | 100% | <98% | Daily |
| Average Prediction Time | <1s | >3s | Weekly |
| Error Count | <10/day | >50/day | Daily |
| Memory Usage | <500MB | >2GB | Weekly |
| Disk Space | >20% free | <10% free | Weekly |

### Tracking Sheet Template

```
Date        | Uptime | Predictions | Errors | Avg Time | Notes
------------|--------|-------------|--------|----------|-------
2025-11-06  | 100%   | 143         | 0      | 0.3s     | Fix deployed
2025-11-07  | 100%   | 89          | 1      | 0.4s     | Minor timeout
...
```

---

## Alert Thresholds

### Critical (Immediate Action)

```bash
# Dashboard Down
# Alert if: curl fails 3 times in a row
# Action: Restart dashboard immediately

# High Error Rate
# Alert if: >50 errors in 1 hour
# Action: Check logs, identify root cause

# Prediction Failures
# Alert if: >10% of predictions fail
# Action: Investigate feature extraction
```

### Warning (Review Within 24h)

```bash
# Slow Predictions
# Alert if: Average >3 seconds
# Action: Check server load, optimize code

# Memory Growth
# Alert if: Memory usage >1GB
# Action: Check for memory leaks

# Disk Space Low
# Alert if: <10% free space
# Action: Clean old logs, archive data
```

### Info (Review Weekly)

```bash
# Model Accuracy Drift
# Alert if: R² drops below 94%
# Action: Consider model retraining

# Usage Patterns
# Alert if: Unusual traffic patterns
# Action: Review logs, verify legitimate use
```

---

## Backup Strategy

### Daily Backups

```bash
# Automated daily backup script
#!/bin/bash
# backup_post_race.sh

BACKUP_DIR="/home/tactical/racing_analytics/backups/post_race"
DATE=$(date +%Y%m%d)

# Backup Python code
tar -czf $BACKUP_DIR/code_$DATE.tar.gz \
    /home/tactical/racing_analytics/src/models/inference/simple_post_race_predictor.py \
    /home/tactical/racing_analytics/src/dashboard/post_race_widget.py

# Backup model file
cp /home/tactical/racing_analytics/data/models/lightgbm_baseline.pkl \
   $BACKUP_DIR/model_$DATE.pkl

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

### Recovery Procedure

```bash
# Restore from backup
cd data_analisys_car

# List available backups
venv/Scripts/python.exe ssh_helper.py "ls -lh /home/tactical/racing_analytics/backups/post_race/"

# Restore specific date
venv/Scripts/python.exe ssh_helper.py "cd /home/tactical/racing_analytics && tar -xzf backups/post_race/code_YYYYMMDD.tar.gz"

# Restart dashboard
venv/Scripts/python.exe ssh_helper.py "pkill -f dashboard && cd /home/tactical/racing_analytics && nohup venv/bin/python src/dashboard/app.py > dashboard.log 2>&1 &"
```

---

## Documentation Updates

### When to Update This Guide

- ✅ After significant code changes
- ✅ When new monitoring tools added
- ✅ After incident resolution (add to troubleshooting)
- ✅ Quarterly review (even if no changes)

### Update Procedure

1. Edit this file locally
2. Update version number and date at top
3. Test any new procedures
4. Commit to repository
5. Notify team of changes

---

## Contact & Escalation

### Level 1: Self-Service
- **Resources:** This guide, TROUBLESHOOTING_GUIDE.md, QUICK_REFERENCE.md
- **Time Limit:** 30 minutes

### Level 2: Team Lead
- **When:** Issue not resolved in 30 minutes
- **Contact:** [Add contact info]
- **Expected Response:** 2 hours

### Level 3: System Administrator
- **When:** Server-level issues, security concerns
- **Contact:** [Add contact info]
- **Expected Response:** 1 hour (critical), 4 hours (non-critical)

---

## Maintenance Schedule

### Daily
- ✅ Check dashboard uptime
- ✅ Review error logs
- ✅ Verify predictions working

### Weekly
- ✅ Run health check script
- ✅ Test with template CSV
- ✅ Review performance metrics

### Monthly
- ✅ Rotate logs
- ✅ Update dependencies (security only)
- ✅ Performance review
- ✅ Backup verification

### Quarterly
- ✅ Full system audit
- ✅ Documentation review
- ✅ Model retraining consideration
- ✅ Architecture review

---

**Last Updated:** 2025-11-06
**Next Review:** 2025-12-06
**Maintained By:** DevOps Team
**Version:** 1.0
