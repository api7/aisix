//! Cache keys for credential-derived upstream tokens.
//!
//! Providers whose credential is an OAuth grant rather than a bearer
//! key (Vertex service accounts, Azure AD app registrations) mint a
//! short-lived token per credential and cache it in-process for its
//! TTL. [`credential_fingerprint`] builds the key those caches use.

use sha2::{Digest, Sha256};

/// Fingerprint a whole upstream credential into an opaque cache key.
///
/// **Pass every field the minter reads.** A key built from the
/// credential's identity fields alone (`client_email`, `client_id`, …)
/// looks correct and silently defeats rotation: those fields are
/// exactly the ones rotation keeps stable, so a rotated credential
/// hits the pre-rotation slot and the gateway keeps authenticating
/// with the replaced secret until the cached token expires — up to an
/// hour after the operator believes the old credential is out of use.
///
/// Include non-secret fields that select the token endpoint
/// (`token_uri`, `authority_host`) too. The contract is that two
/// credentials with the same fingerprint are interchangeable, which is
/// what lets provider keys backed by one credential share a token
/// slot.
pub fn credential_fingerprint(fields: &[&str]) -> String {
    let mut hasher = Sha256::new();
    // Length-prefix each field: without it ("ab", "c") and ("a", "bc")
    // hash alike, so one credential could adopt another's token slot.
    for field in fields {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_any_field_changes_the_fingerprint() {
        let before = credential_fingerprint(&["sa@example.com", "-----BEGIN old-key"]);
        let after = credential_fingerprint(&["sa@example.com", "-----BEGIN new-key"]);
        assert_ne!(
            before, after,
            "a rotated secret under an unchanged identity must not reuse the cache slot"
        );
    }

    #[test]
    fn identical_credentials_share_one_slot() {
        assert_eq!(
            credential_fingerprint(&["tenant", "app", "secret"]),
            credential_fingerprint(&["tenant", "app", "secret"]),
        );
    }

    #[test]
    fn field_boundaries_are_unambiguous() {
        assert_ne!(
            credential_fingerprint(&["ab", "c"]),
            credential_fingerprint(&["a", "bc"]),
        );
    }

    #[test]
    fn empty_field_is_distinct_from_absent_field() {
        assert_ne!(
            credential_fingerprint(&["tenant", ""]),
            credential_fingerprint(&["tenant"]),
        );
    }
}
