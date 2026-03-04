pub fn format_jsonschema_error(evaluation: &jsonschema::Evaluation) -> String {
    evaluation
        .iter_errors()
        .map(|err| {
            let path = err.instance_location.as_str();
            format!(
                "property \"{}\" validation failed: {}",
                if path.is_empty() { "/" } else { path },
                err.error,
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
