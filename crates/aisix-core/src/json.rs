use serde::Serialize;
use serde_json::Value;

/// `value` as a [`Value`], each number read as the text it serializes to.
///
/// [`serde_json::to_value`] widens an `f32` to `f64`, so a configured `0.9`
/// would come out as `0.8999999761581421` wherever the `Value` is written.
/// Serializing to text and parsing that back yields the number the field
/// holds.
pub fn to_value<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<Value> {
    serde_json::from_slice(&serde_json::to_vec(value)?)
}

#[cfg(test)]
mod tests {
    #[test]
    fn an_f32_keeps_its_decimal() {
        assert_eq!(super::to_value(&0.9_f32).unwrap().to_string(), "0.9");
        assert_eq!(
            super::to_value(&[Some(0.9123_f32)]).unwrap().to_string(),
            "[0.9123]"
        );
    }
}
