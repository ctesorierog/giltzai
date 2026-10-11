-- GiltzAI QA views. Every view must return 0 rows (flAIre rule).
CREATE VIEW qa_listing_dup_ids AS SELECT listing_id FROM listings GROUP BY listing_id HAVING COUNT(*) > 1;
CREATE VIEW qa_listing_no_zone AS SELECT listing_id FROM listings WHERE zone_code IS NULL;
CREATE VIEW qa_listing_zone_orphan AS SELECT lz.listing_id FROM listing_zone lz LEFT JOIN listings l ON l.listing_id = lz.listing_id WHERE l.listing_id IS NULL;
CREATE VIEW qa_listing_zone_bad_code AS SELECT lz.listing_id FROM listing_zone lz LEFT JOIN zones z ON z.zone_code = lz.zone_code WHERE z.zone_code IS NULL;
CREATE VIEW qa_listing_outside_donostia_bbox AS SELECT listing_id FROM listings WHERE latitude NOT BETWEEN 43.24 AND 43.34 OR longitude NOT BETWEEN -2.10 AND -1.90;
CREATE VIEW qa_listing_not_in_raw AS SELECT l.listing_id FROM listings l LEFT JOIN listings_raw r ON r.id = l.listing_id WHERE r.id IS NULL;
CREATE VIEW qa_listing_host_not_hashed AS SELECT listing_id FROM listings WHERE host_hash IS NULL OR length(host_hash) <> 16;
CREATE VIEW qa_listing_code_not_normalised AS SELECT listing_id FROM listings WHERE code_best IS NOT NULL AND code_best NOT GLOB '[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][0-9]';
CREATE VIEW qa_register_dup_ids AS SELECT register_id FROM register GROUP BY register_id HAVING COUNT(*) > 1;
CREATE VIEW qa_register_code_not_normalised AS SELECT register_id FROM register WHERE code_norm IS NULL OR code_norm NOT GLOB '[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][0-9]' OR code_norm <> register_id;
CREATE VIEW qa_register_outside_donostia AS SELECT register_id FROM register WHERE lat IS NOT NULL AND (lat NOT BETWEEN 43.24 AND 43.34 OR lon NOT BETWEEN -2.10 AND -1.90);
CREATE VIEW qa_code_candidates_not_normalised AS SELECT cand_id FROM code_candidates WHERE code_norm IS NOT NULL AND code_norm NOT GLOB '[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][0-9]';
CREATE VIEW qa_code_candidates_orphan AS SELECT c.cand_id FROM code_candidates c LEFT JOIN listings l ON l.listing_id = c.listing_id WHERE l.listing_id IS NULL;
CREATE VIEW qa_match_candidates_orphan AS SELECT m.listing_id FROM match_candidates m LEFT JOIN listings l ON l.listing_id = m.listing_id LEFT JOIN register r ON r.register_id = m.register_id WHERE l.listing_id IS NULL OR r.register_id IS NULL;
CREATE VIEW qa_labels_orphan AS SELECT lb.listing_id FROM labels lb LEFT JOIN listings l ON l.listing_id = lb.listing_id WHERE l.listing_id IS NULL;
CREATE VIEW qa_labels_dup AS SELECT listing_id FROM labels GROUP BY listing_id HAVING COUNT(*) > 1;
CREATE VIEW qa_labels_not_150 AS SELECT 1 WHERE (SELECT COUNT(*) FROM labels) <> 150;
CREATE VIEW qa_baseline_sum_mismatch AS SELECT 1 WHERE (SELECT SUM(listings_all) FROM zone_baseline) <> (SELECT COUNT(*) FROM listings);
CREATE VIEW qa_personal_columns AS
  SELECT 'listings_raw.' || name FROM pragma_table_info('listings_raw') WHERE name IN ('host_name','host_id','host_url','listing_url','picture_url','host_about','host_picture_url','host_thumbnail_url','host_profile_url')
  UNION ALL SELECT 'listings.' || name FROM pragma_table_info('listings') WHERE name IN ('host_name','host_id','listing_url','description','name')
  UNION ALL SELECT 'register_raw.' || name FROM pragma_table_info('register_raw') WHERE name IN ('Teléfono','Email','Web','Nombre comercial');
