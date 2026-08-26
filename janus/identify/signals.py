"""Canonical signal families: the bridge from Pillar 1 to Pillar 3.

Attack cards describe what a defender could *see* in prose-like observable names. Those names
are deliberately specific - "collect_request_burst_to_many_vpas" says more than "velocity" -
but specificity alone does not build features. Each observable therefore declares a
:class:`SignalFamily`, and it is the family that the feature store implements.

That indirection is what makes the pipeline traceable in both directions: every feature in
``janus.defend.features`` exists because some attack card asked for it, and every card can be
asked which of its observables the current defence actually computes. The atlas cannot quietly
describe a signal that nothing detects, and the defence cannot quietly grow features that no
known attack motivates.

Generated from the atlas (191 observables, 24 families) and frozen here so
the mapping is reviewable in diff rather than recomputed by fragile keyword rules at runtime.
"""

from __future__ import annotations

from enum import StrEnum


class SignalFamily(StrEnum):
    """What a detection feature is fundamentally measuring."""

    AGENT_ACTION = "agent_action"
    AMOUNT_ANOMALY = "amount_anomaly"
    ARTIFACT_FORENSICS = "artifact_forensics"
    AUTH_ANOMALY = "auth_anomaly"
    BALANCE_SWEEP = "balance_sweep"
    BENEFICIARY_NOVELTY = "beneficiary_novelty"
    CONTROL_BYPASS = "control_bypass"
    CREDENTIAL_CHANGE = "credential_change"
    DECLINE_PROBING = "decline_probing"
    DEVICE_NOVELTY = "device_novelty"
    DORMANCY_BREAK = "dormancy_break"
    ESCALATION_SEQUENCE = "escalation_sequence"
    GEO_ANOMALY = "geo_anomaly"
    GRAPH_FANIN_FANOUT = "graph_fanin_fanout"
    GRAPH_PASSTHROUGH = "graph_passthrough"
    IDENTITY_LINKAGE = "identity_linkage"
    MERCHANT_DISPUTE = "merchant_dispute"
    MERCHANT_LIFECYCLE = "merchant_lifecycle"
    MODEL_INTEGRITY = "model_integrity"
    SESSION_CONTEXT = "session_context"
    TEMPORAL_REGULARITY = "temporal_regularity"
    TENURE_MISMATCH = "tenure_mismatch"
    TEXT_CONTENT = "text_content"
    THRESHOLD_HUGGING = "threshold_hugging"


FAMILY_DESCRIPTIONS: dict[SignalFamily, str] = {
    SignalFamily.AGENT_ACTION: "An AI agent, not a human, took the action that moved money.",
    SignalFamily.AMOUNT_ANOMALY: "Value is out of pattern for this entity or population.",
    SignalFamily.ARTIFACT_FORENSICS: "The supporting media or biometric artefact is itself synthetic.",
    SignalFamily.AUTH_ANOMALY: "Authentication succeeded but its context contradicts the claim.",
    SignalFamily.BALANCE_SWEEP: "Available value is being emptied rather than spent.",
    SignalFamily.BENEFICIARY_NOVELTY: "Money is moving to a counterparty with no prior relationship.",
    SignalFamily.CONTROL_BYPASS: "A required control was skipped, overridden or never engaged.",
    SignalFamily.CREDENTIAL_CHANGE: "Contact details or device bindings changed before the payment.",
    SignalFamily.DECLINE_PROBING: "The pattern is a search for what works, not an attempt to buy.",
    SignalFamily.DEVICE_NOVELTY: "The device is unknown, emulated, or remotely controlled.",
    SignalFamily.DORMANCY_BREAK: "A quiet account woke up and immediately carried volume.",
    SignalFamily.ESCALATION_SEQUENCE: "Amounts or trust ratchet upward across a session or campaign.",
    SignalFamily.GEO_ANOMALY: "Location is inconsistent with the entity's history or with itself.",
    SignalFamily.GRAPH_FANIN_FANOUT: "Many-to-one or one-to-many structure in the transfer graph.",
    SignalFamily.GRAPH_PASSTHROUGH: "Value flows through without resting; the account is a pipe.",
    SignalFamily.IDENTITY_LINKAGE: "Nominally unrelated identities share more than chance allows.",
    SignalFamily.MERCHANT_DISPUTE: "Post-transaction reversal behaviour is out of pattern.",
    SignalFamily.MERCHANT_LIFECYCLE: "The acceptance point is too new, too fast, or too short-lived.",
    SignalFamily.MODEL_INTEGRITY: "The defence itself is being probed, drifted or poisoned.",
    SignalFamily.SESSION_CONTEXT: "Out-of-band context around the payment changes its meaning.",
    SignalFamily.TEMPORAL_REGULARITY: "Timing is machine-like, or wrong for this human.",
    SignalFamily.TENURE_MISMATCH: "Claimed history and observed history do not agree.",
    SignalFamily.TEXT_CONTENT: "Free-text fields carry the semantics of a scam.",
    SignalFamily.THRESHOLD_HUGGING: "Behaviour is shaped to sit just under a known control limit.",
}


#: Every observable used anywhere in the atlas, mapped to the family a feature would implement.
OBSERVABLE_SIGNALS: dict[str, SignalFamily] = {
    # AGENT_ACTION - An AI agent, not a human, took the action that moved money.
    "agent_initiated_purchase_flag": SignalFamily.AGENT_ACTION,
    "agent_issued_payout_without_matching_order": SignalFamily.AGENT_ACTION,
    "agent_tool_call_parameters_outside_policy": SignalFamily.AGENT_ACTION,
    "conversation_turn_count_before_refund_anomalous": SignalFamily.AGENT_ACTION,
    "instruction_like_text_in_customer_supplied_fields": SignalFamily.AGENT_ACTION,
    "merchant_selected_never_browsed_by_human": SignalFamily.AGENT_ACTION,
    "purchase_without_human_session_activity": SignalFamily.AGENT_ACTION,
    # AMOUNT_ANOMALY - Value is out of pattern for this entity or population.
    "payment_amount_matches_utility_bill_distribution": SignalFamily.AMOUNT_ANOMALY,
    "payment_amounts_match_retail_distribution": SignalFamily.AMOUNT_ANOMALY,
    "payment_velocity_spike_vs_personal_baseline": SignalFamily.AMOUNT_ANOMALY,
    # ARTIFACT_FORENSICS - The supporting media or biometric artefact is itself synthetic.
    "behavioural_score_high_but_device_unseen": SignalFamily.ARTIFACT_FORENSICS,
    "biometric_pass_conflicts_with_network_signals": SignalFamily.ARTIFACT_FORENSICS,
    "dispute_narrative_style_anomaly": SignalFamily.ARTIFACT_FORENSICS,
    "document_image_generative_artifacts": SignalFamily.ARTIFACT_FORENSICS,
    "document_metadata_anomalies": SignalFamily.ARTIFACT_FORENSICS,
    "evidence_image_generative_artifacts": SignalFamily.ARTIFACT_FORENSICS,
    "kyb_document_metadata_anomalies": SignalFamily.ARTIFACT_FORENSICS,
    "kyc_liveness_confidence_borderline": SignalFamily.ARTIFACT_FORENSICS,
    "liveness_confidence_borderline_but_passing": SignalFamily.ARTIFACT_FORENSICS,
    "trace_entropy_lower_than_human_baseline": SignalFamily.ARTIFACT_FORENSICS,
    "virtual_camera_or_emulator_signals": SignalFamily.ARTIFACT_FORENSICS,
    # AUTH_ANOMALY - Authentication succeeded but its context contradicts the claim.
    "authenticated_transaction_from_unseen_device": SignalFamily.AUTH_ANOMALY,
    "challenge_response_latency_atypical": SignalFamily.AUTH_ANOMALY,
    "contactless_spend_ramp_immediately_post_provisioning": SignalFamily.AUTH_ANOMALY,
    "login_success_from_unseen_device_and_geo": SignalFamily.AUTH_ANOMALY,
    "otp_consumed_from_different_ip_than_session": SignalFamily.AUTH_ANOMALY,
    "otp_entered_within_seconds_of_sms": SignalFamily.AUTH_ANOMALY,
    "provisioning_geo_differs_from_card_history": SignalFamily.AUTH_ANOMALY,
    "sim_swap_recently_flagged_for_msisdn": SignalFamily.AUTH_ANOMALY,
    "step_up_passed_from_unseen_device": SignalFamily.AUTH_ANOMALY,
    "three_ds_challenge_latency_atypical": SignalFamily.AUTH_ANOMALY,
    "wallet_provisioning_step_up_via_contact_centre": SignalFamily.AUTH_ANOMALY,
    # BALANCE_SWEEP - Available value is being emptied rather than spent.
    "cash_advance_spike_atypical_for_profile": SignalFamily.BALANCE_SWEEP,
    "full_balance_sweep": SignalFamily.BALANCE_SWEEP,
    "lite_balance_topup_then_immediate_drain": SignalFamily.BALANCE_SWEEP,
    "new_loan_drawdown_then_immediate_transfer": SignalFamily.BALANCE_SWEEP,
    "savings_product_liquidation_precedes_transfer": SignalFamily.BALANCE_SWEEP,
    "simultaneous_max_drawdown_across_products": SignalFamily.BALANCE_SWEEP,
    "stored_value_drain_shortly_after_login": SignalFamily.BALANCE_SWEEP,
    "utilisation_jump_after_long_low_utilisation": SignalFamily.BALANCE_SWEEP,
    # BENEFICIARY_NOVELTY - Money is moving to a counterparty with no prior relationship.
    "aeps_withdrawal_without_prior_aeps_history": SignalFamily.BENEFICIARY_NOVELTY,
    "beneficiary_add_then_immediate_high_value_transfer": SignalFamily.BENEFICIARY_NOVELTY,
    "biller_payment_to_non_verified_handle": SignalFamily.BENEFICIARY_NOVELTY,
    "cnp_at_never_seen_merchant": SignalFamily.BENEFICIARY_NOVELTY,
    "collect_request_burst_to_many_vpas": SignalFamily.BENEFICIARY_NOVELTY,
    "collect_request_from_unknown_payee": SignalFamily.BENEFICIARY_NOVELTY,
    "counterparty_known_crypto_desk": SignalFamily.BENEFICIARY_NOVELTY,
    "counterparty_set_entirely_novel": SignalFamily.BENEFICIARY_NOVELTY,
    "digital_goods_high_value_first_purchase": SignalFamily.BENEFICIARY_NOVELTY,
    "first_time_beneficiary": SignalFamily.BENEFICIARY_NOVELTY,
    "high_value_first_purchase_at_merchant": SignalFamily.BENEFICIARY_NOVELTY,
    "new_corporate_beneficiary": SignalFamily.BENEFICIARY_NOVELTY,
    "payee_vpa_string_distance_near_known_biller": SignalFamily.BENEFICIARY_NOVELTY,
    "payer_has_no_prior_relationship_with_requester": SignalFamily.BENEFICIARY_NOVELTY,
    "requester_collect_success_ratio_low": SignalFamily.BENEFICIARY_NOVELTY,
    "same_beneficiary_across_unrelated_support_cases": SignalFamily.BENEFICIARY_NOVELTY,
    "same_payee_repeated_small_amounts": SignalFamily.BENEFICIARY_NOVELTY,
    # CONTROL_BYPASS - A required control was skipped, overridden or never engaged.
    "approval_chain_shortcut": SignalFamily.CONTROL_BYPASS,
    "invoice_unit_price_deviates_from_market_comparable": SignalFamily.CONTROL_BYPASS,
    "payment_exceeds_role_historical_maximum": SignalFamily.CONTROL_BYPASS,
    "refund_approved_by_automated_agent_above_policy_limit": SignalFamily.CONTROL_BYPASS,
    "refund_issued_without_return_scan": SignalFamily.CONTROL_BYPASS,
    "trade_flow_without_corresponding_logistics_records": SignalFamily.CONTROL_BYPASS,
    # CREDENTIAL_CHANGE - Contact details or device bindings changed before the payment.
    "address_change_precedes_card_reissue": SignalFamily.CREDENTIAL_CHANGE,
    "channel_switch_before_payment": SignalFamily.CREDENTIAL_CHANGE,
    "contact_detail_change_precedes_beneficiary_add": SignalFamily.CREDENTIAL_CHANGE,
    "credential_reset_then_immediate_high_value_transfer": SignalFamily.CREDENTIAL_CHANGE,
    "expedited_delivery_requested": SignalFamily.CREDENTIAL_CHANGE,
    "inbound_sms_channel_changed_recently": SignalFamily.CREDENTIAL_CHANGE,
    "notification_channel_recently_modified": SignalFamily.CREDENTIAL_CHANGE,
    "reissue_then_atm_withdrawal_in_new_geography": SignalFamily.CREDENTIAL_CHANGE,
    "servicing_change_outside_customer_session": SignalFamily.CREDENTIAL_CHANGE,
    # DECLINE_PROBING - The pattern is a search for what works, not an attempt to buy.
    "cross_merchant_correlated_decline_pattern": SignalFamily.DECLINE_PROBING,
    "decline_ratio_high_without_monetisation_attempt": SignalFamily.DECLINE_PROBING,
    "high_decline_ratio_per_device": SignalFamily.DECLINE_PROBING,
    "low_volume_high_success_credential_attempts": SignalFamily.DECLINE_PROBING,
    "luhn_valid_but_never_issued_pans": SignalFamily.DECLINE_PROBING,
    "merchant_approval_rate_sudden_drop": SignalFamily.DECLINE_PROBING,
    "micro_amount_authorisation_burst": SignalFamily.DECLINE_PROBING,
    "probe_like_low_value_transactions_precede_attack": SignalFamily.DECLINE_PROBING,
    "rapid_successive_auth_attempts": SignalFamily.DECLINE_PROBING,
    "sequential_or_clustered_bin_ranges": SignalFamily.DECLINE_PROBING,
    "single_entity_high_variance_low_value_attempts": SignalFamily.DECLINE_PROBING,
    "zero_value_auth_spike_at_merchant": SignalFamily.DECLINE_PROBING,
    # DEVICE_NOVELTY - The device is unknown, emulated, or remotely controlled.
    "accessibility_service_active_during_payment": SignalFamily.DEVICE_NOVELTY,
    "device_fingerprint_unseen_for_card": SignalFamily.DEVICE_NOVELTY,
    "device_id_never_seen_for_cardholder": SignalFamily.DEVICE_NOVELTY,
    "remote_access_app_concurrent_with_session": SignalFamily.DEVICE_NOVELTY,
    # DORMANCY_BREAK - A quiet account woke up and immediately carried volume.
    "account_dormant_then_sudden_high_throughput": SignalFamily.DORMANCY_BREAK,
    "dormancy_period_ends_with_high_value_activity": SignalFamily.DORMANCY_BREAK,
    "mandate_created_then_dormant_then_large_debit": SignalFamily.DORMANCY_BREAK,
    # ESCALATION_SEQUENCE - Amounts or trust ratchet upward across a session or campaign.
    "amount_escalation_within_session": SignalFamily.ESCALATION_SEQUENCE,
    "escalating_deposit_sequence": SignalFamily.ESCALATION_SEQUENCE,
    "purchase_then_immediate_intra_platform_transfer": SignalFamily.ESCALATION_SEQUENCE,
    "reciprocal_low_value_then_high_value_pattern": SignalFamily.ESCALATION_SEQUENCE,
    "small_inbound_credit_precedes_larger_outbound": SignalFamily.ESCALATION_SEQUENCE,
    "sustained_escalating_payments_to_same_counterparty": SignalFamily.ESCALATION_SEQUENCE,
    "token_provisioned_to_new_device_then_immediate_spend": SignalFamily.ESCALATION_SEQUENCE,
    # GEO_ANOMALY - Location is inconsistent with the entity's history or with itself.
    "cross_border_first_use": SignalFamily.GEO_ANOMALY,
    "payee_geo_cluster_but_personal_account_type": SignalFamily.GEO_ANOMALY,
    "shipping_billing_mismatch": SignalFamily.GEO_ANOMALY,
    "victim_geo_far_from_terminal_geo": SignalFamily.GEO_ANOMALY,
    # GRAPH_FANIN_FANOUT - Many-to-one or one-to-many structure in the transfer graph.
    "beneficiary_fan_in_from_unrelated_payers": SignalFamily.GRAPH_FANIN_FANOUT,
    "closed_loop_value_transfer_between_new_accounts": SignalFamily.GRAPH_FANIN_FANOUT,
    "counterparty_receives_from_many_unrelated_payers": SignalFamily.GRAPH_FANIN_FANOUT,
    "cross_institution_chain_within_short_window": SignalFamily.GRAPH_FANIN_FANOUT,
    "fan_in_then_fan_out_within_short_window": SignalFamily.GRAPH_FANIN_FANOUT,
    "hop_depth_exceeds_typical_p2p_pattern": SignalFamily.GRAPH_FANIN_FANOUT,
    "intra_cohort_transaction_ratio_high": SignalFamily.GRAPH_FANIN_FANOUT,
    "many_unrelated_payers_to_new_personal_vpa": SignalFamily.GRAPH_FANIN_FANOUT,
    "many_unrelated_payers_to_single_counterparty": SignalFamily.GRAPH_FANIN_FANOUT,
    "many_unrelated_payers_to_single_merchant": SignalFamily.GRAPH_FANIN_FANOUT,
    "many_unrelated_victims_single_operator": SignalFamily.GRAPH_FANIN_FANOUT,
    "transfer_graph_component_size_anomaly": SignalFamily.GRAPH_FANIN_FANOUT,
    # GRAPH_PASSTHROUGH - Value flows through without resting; the account is a pipe.
    "account_becomes_pass_through_shortly_after_dormancy": SignalFamily.GRAPH_PASSTHROUGH,
    "aggregate_throughput_high_despite_small_transactions": SignalFamily.GRAPH_PASSTHROUGH,
    "balance_retention_ratio_near_zero": SignalFamily.GRAPH_PASSTHROUGH,
    "counterparty_set_churns_rapidly": SignalFamily.GRAPH_PASSTHROUGH,
    "device_rebinding_precedes_throughput_spike": SignalFamily.GRAPH_PASSTHROUGH,
    "grooming_period_then_throughput_spike": SignalFamily.GRAPH_PASSTHROUGH,
    "short_dwell_time_before_onward_transfer": SignalFamily.GRAPH_PASSTHROUGH,
    # IDENTITY_LINKAGE - Nominally unrelated identities share more than chance allows.
    "address_or_device_shared_across_identities": SignalFamily.IDENTITY_LINKAGE,
    "cohort_onboarded_within_narrow_window": SignalFamily.IDENTITY_LINKAGE,
    "counterparties_share_beneficial_ownership_signals": SignalFamily.IDENTITY_LINKAGE,
    "identity_cluster_shares_attributes_beyond_chance": SignalFamily.IDENTITY_LINKAGE,
    "many_distinct_pans_single_device_fingerprint": SignalFamily.IDENTITY_LINKAGE,
    "many_signups_shared_device_or_ip_cluster": SignalFamily.IDENTITY_LINKAGE,
    "new_mule_accounts_share_onboarding_pattern": SignalFamily.IDENTITY_LINKAGE,
    "onboarding_device_shared_across_identities": SignalFamily.IDENTITY_LINKAGE,
    "recruitment_cohort_activates_synchronously": SignalFamily.IDENTITY_LINKAGE,
    "staff_id_associated_with_multiple_flagged_records": SignalFamily.IDENTITY_LINKAGE,
    "virtual_card_bin_concentration": SignalFamily.IDENTITY_LINKAGE,
    "withdrawals_cluster_at_single_agent_terminal": SignalFamily.IDENTITY_LINKAGE,
    # MERCHANT_DISPUTE - Post-transaction reversal behaviour is out of pattern.
    "customer_refund_ratio_elevated": SignalFamily.MERCHANT_DISPUTE,
    "delivery_confirmed_but_disputed_not_received": SignalFamily.MERCHANT_DISPUTE,
    "dispute_rate_per_cardholder_elevated": SignalFamily.MERCHANT_DISPUTE,
    "disputes_concentrated_in_high_resale_categories": SignalFamily.MERCHANT_DISPUTE,
    "gift_card_or_digital_goods_concentration": SignalFamily.MERCHANT_DISPUTE,
    "merchant_dispute_ratio_rising": SignalFamily.MERCHANT_DISPUTE,
    "merchant_refund_rate_near_zero_with_high_disputes": SignalFamily.MERCHANT_DISPUTE,
    "refund_amount_exceeds_original_capture": SignalFamily.MERCHANT_DISPUTE,
    "secondary_marketplace_resale_pattern": SignalFamily.MERCHANT_DISPUTE,
    # MERCHANT_LIFECYCLE - The acceptance point is too new, too fast, or too short-lived.
    "mandate_revocation_rate_high_for_merchant": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_age_days_low": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_category_code_mismatch": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_category_mismatch_vs_behaviour": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_mandate_portfolio_grows_abnormally_fast": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_payer_base_geographically_dispersed": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_transaction_volume_sudden_drop": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_volume_ramp_steep": SignalFamily.MERCHANT_LIFECYCLE,
    "merchant_vpa_short_lifetime_high_volume": SignalFamily.MERCHANT_LIFECYCLE,
    "payments_to_recently_registered_merchant": SignalFamily.MERCHANT_LIFECYCLE,
    "promotional_credit_redemption_velocity": SignalFamily.MERCHANT_LIFECYCLE,
    "settlement_withdrawal_velocity_high": SignalFamily.MERCHANT_LIFECYCLE,
    "signup_to_cancel_lifecycle_uniform": SignalFamily.MERCHANT_LIFECYCLE,
    # MODEL_INTEGRITY - The defence itself is being probed, drifted or poisoned.
    "approval_rate_for_cohort_anomalously_high": SignalFamily.MODEL_INTEGRITY,
    "attack_features_cluster_just_below_decision_threshold": SignalFamily.MODEL_INTEGRITY,
    "attempts_span_feature_space_unnaturally_evenly": SignalFamily.MODEL_INTEGRITY,
    "cohort_with_anomalously_consistent_benign_outcomes": SignalFamily.MODEL_INTEGRITY,
    "feature_importance_shift_across_model_versions": SignalFamily.MODEL_INTEGRITY,
    "label_distribution_drift_between_retrains": SignalFamily.MODEL_INTEGRITY,
    "post_retrain_recall_drop_for_specific_family": SignalFamily.MODEL_INTEGRITY,
    "score_distribution_of_fraud_shifts_downward_over_time": SignalFamily.MODEL_INTEGRITY,
    "systematic_feature_sweep_pattern_across_attempts": SignalFamily.MODEL_INTEGRITY,
    "unusual_uniformity_in_controllable_attributes": SignalFamily.MODEL_INTEGRITY,
    # SESSION_CONTEXT - Out-of-band context around the payment changes its meaning.
    "inbound_call_precedes_payment": SignalFamily.SESSION_CONTEXT,
    "session_behaviour_diverges_from_user_baseline": SignalFamily.SESSION_CONTEXT,
    # TEMPORAL_REGULARITY - Timing is machine-like, or wrong for this human.
    "collect_approved_within_seconds": SignalFamily.TEMPORAL_REGULARITY,
    "debit_interarrival_time_machine_regular": SignalFamily.TEMPORAL_REGULARITY,
    "interaction_timing_atypical_for_user": SignalFamily.TEMPORAL_REGULARITY,
    "interarrival_time_machine_regular": SignalFamily.TEMPORAL_REGULARITY,
    "out_of_pattern_hour": SignalFamily.TEMPORAL_REGULARITY,
    "purchase_cadence_machine_regular": SignalFamily.TEMPORAL_REGULARITY,
    "transfer_pattern_persists_over_weeks": SignalFamily.TEMPORAL_REGULARITY,
    # TENURE_MISMATCH - Claimed history and observed history do not agree.
    "beneficiary_account_age_days_low": SignalFamily.TENURE_MISMATCH,
    "entity_age_days_low_with_high_value_trade": SignalFamily.TENURE_MISMATCH,
    "identity_attributes_never_seen_in_bureau": SignalFamily.TENURE_MISMATCH,
    "identity_attributes_synthetic_pattern": SignalFamily.TENURE_MISMATCH,
    "identity_thin_file_at_origination": SignalFamily.TENURE_MISMATCH,
    "payee_vpa_age_days_low": SignalFamily.TENURE_MISMATCH,
    "repayment_behaviour_too_regular": SignalFamily.TENURE_MISMATCH,
    "tenure_high_but_behavioural_history_sparse": SignalFamily.TENURE_MISMATCH,
    "young_account_holder_atypical_volume": SignalFamily.TENURE_MISMATCH,
    # TEXT_CONTENT - Free-text fields carry the semantics of a scam.
    "payment_reference_investment_keywords": SignalFamily.TEXT_CONTENT,
    "payment_reference_safe_account_keywords": SignalFamily.TEXT_CONTENT,
    "urgency_keywords_in_payment_reference": SignalFamily.TEXT_CONTENT,
    # THRESHOLD_HUGGING - Behaviour is shaped to sit just under a known control limit.
    "amount_split_ratios_non_round": SignalFamily.THRESHOLD_HUGGING,
    "debit_amount_just_below_notification_threshold": SignalFamily.THRESHOLD_HUGGING,
    "distributed_low_count_attempts_same_bin": SignalFamily.THRESHOLD_HUGGING,
    "mandate_ceiling_far_exceeds_first_debit": SignalFamily.THRESHOLD_HUGGING,
    "repeated_sub_threshold_debits": SignalFamily.THRESHOLD_HUGGING,
}


def family_of(observable: str) -> SignalFamily:
    """Look up an observable's signal family, failing loudly on an unregistered name."""
    try:
        return OBSERVABLE_SIGNALS[observable]
    except KeyError as exc:
        raise KeyError(
            f"observable {observable!r} is not in the signal registry; "
            "add it to janus/identify/signals.py so a feature can be held responsible for it"
        ) from exc


def observables_for(family: SignalFamily) -> list[str]:
    """Every atlas observable that a given family is responsible for detecting."""
    return sorted(o for o, f in OBSERVABLE_SIGNALS.items() if f is family)
