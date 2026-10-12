# E-Commerce Customer Segmentation Report

## Executive Summary
Using unsupervised K-Means clustering combined with engineered RFM features, our customer base was segmented into four distinct behavioral groups. This allows targeted marketing without relying on pre-existing target labels.

## Cluster Profiles
1. **Cluster 0: Champions / High-Value Loyalists**
   - *Characteristics*: High total spend, frequent purchases, low recency days.
   - *Business Action*: Offer exclusive VIP perks and early product access.
2. **Cluster 1: At-Risk / Churning Customers**
   - *Characteristics*: High past spend, but very high recency (long time since last purchase).
   - *Business Action*: Deploy re-engagement email campaigns and win-back discounts.
3. **Cluster 2: New / Low-Engagement Shoppers**
   - *Characteristics*: Low order count, minimal spend, recent initial activity.
   - *Business Action*: Provide onboarding tutorials and first-purchase follow-up offers.
4. **Cluster 3: Occasional Bargain Hunters**
   - *Characteristics*: Moderate frequency, low average order value, price-sensitive.
   - *Business Action*: Promote bundled discounts and clearance sales.

## Limitations & Stability
- Cluster labels are mathematical representations based on distance metrics and should not be treated as absolute ground truth.
- 2D PCA visualization captures approximately 78% of total dataset variance; minor overlaps in 2D space are expected due to dimensionality reduction.