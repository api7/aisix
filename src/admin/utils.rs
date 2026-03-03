pub fn format_jsonschema_error(evaluation: &jsonschema::Evaluation) -> String {
    evaluation
        .iter_errors()
        .map(|err| {
            let path = err.instance_location.as_str();
            format!(
                "property \"{}\" validation failed: {}",
                path.is_empty().then(|| "/").unwrap_or(path),
                err.error,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
