//! A `serde::Serializer` that only measures: the compact-JSON length of a
//! value, without escaping or writing anything, stopping as soon as the
//! running total passes a limit. A string counts its unescaped bytes plus
//! its quotes, so a long string costs one addition rather than a scan.

use serde::ser::{self, Serialize};

/// Generous upper bound on a number's (or a char's) JSON text.
const SCALAR: usize = 24;

pub(super) struct JsonLen {
    total: usize,
    limit: usize,
}

#[derive(Debug)]
pub(super) struct OverLimit;

impl std::fmt::Display for OverLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("serialized size over the limit")
    }
}

impl std::error::Error for OverLimit {}

impl ser::Error for OverLimit {
    fn custom<T: std::fmt::Display>(_: T) -> Self {
        OverLimit
    }
}

/// The compact-JSON length of `value`, or `None` once it passes `limit`
/// (or the value refuses to serialize).
pub(super) fn json_len_within(value: &impl Serialize, limit: usize) -> Option<usize> {
    let mut len = JsonLen { total: 0, limit };
    value.serialize(&mut len).ok().map(|()| len.total)
}

impl JsonLen {
    fn add(&mut self, n: usize) -> Result<(), OverLimit> {
        self.total = self.total.saturating_add(n);
        if self.total > self.limit {
            Err(OverLimit)
        } else {
            Ok(())
        }
    }
}

impl ser::Serializer for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_bool(self, _: bool) -> Result<(), OverLimit> {
        self.add(5)
    }
    fn serialize_i8(self, _: i8) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_i16(self, _: i16) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_i32(self, _: i32) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_i64(self, _: i64) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_i128(self, _: i128) -> Result<(), OverLimit> {
        self.add(2 * SCALAR)
    }
    fn serialize_u8(self, _: u8) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_u16(self, _: u16) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_u32(self, _: u32) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_u64(self, _: u64) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_u128(self, _: u128) -> Result<(), OverLimit> {
        self.add(2 * SCALAR)
    }
    fn serialize_f32(self, _: f32) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_f64(self, _: f64) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_char(self, _: char) -> Result<(), OverLimit> {
        self.add(SCALAR)
    }
    fn serialize_str(self, v: &str) -> Result<(), OverLimit> {
        self.add(v.len().saturating_add(2))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<(), OverLimit> {
        // serde_json writes bytes as an array of numbers: up to "255,".
        self.add(v.len().saturating_mul(4).saturating_add(2))
    }
    fn serialize_none(self) -> Result<(), OverLimit> {
        self.add(4)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), OverLimit> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), OverLimit> {
        self.add(4)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), OverLimit> {
        self.add(4)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<(), OverLimit> {
        self.serialize_str(variant)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        value: &T,
    ) -> Result<(), OverLimit> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<(), OverLimit> {
        // {"variant":value}
        self.add(variant.len().saturating_add(5))?;
        value.serialize(self)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self, OverLimit> {
        self.add(2)?;
        Ok(self)
    }
    fn serialize_tuple(self, _: usize) -> Result<Self, OverLimit> {
        self.add(2)?;
        Ok(self)
    }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, OverLimit> {
        self.add(2)?;
        Ok(self)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Self, OverLimit> {
        // {"variant":[...]}
        self.add(variant.len().saturating_add(7))?;
        Ok(self)
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Self, OverLimit> {
        self.add(2)?;
        Ok(self)
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, OverLimit> {
        self.add(2)?;
        Ok(self)
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Self, OverLimit> {
        // {"variant":{...}}
        self.add(variant.len().saturating_add(7))?;
        Ok(self)
    }
}

/// One element of a sequence: its value plus a separating comma.
fn element<T: ?Sized + Serialize>(len: &mut JsonLen, value: &T) -> Result<(), OverLimit> {
    len.add(1)?;
    value.serialize(len)
}

/// One field of an object: `"key":value,`.
fn field<T: ?Sized + Serialize>(len: &mut JsonLen, key: &str, value: &T) -> Result<(), OverLimit> {
    len.add(key.len().saturating_add(4))?;
    value.serialize(len)
}

impl ser::SerializeSeq for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), OverLimit> {
        element(self, value)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

impl ser::SerializeTuple for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), OverLimit> {
        element(self, value)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

impl ser::SerializeTupleStruct for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), OverLimit> {
        element(self, value)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

impl ser::SerializeTupleVariant for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), OverLimit> {
        element(self, value)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

impl ser::SerializeMap for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), OverLimit> {
        // The key, its colon and the separating comma; a non-string key
        // gains quotes in serde_json, which `SCALAR` already covers.
        self.add(2)?;
        key.serialize(&mut **self)
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), OverLimit> {
        value.serialize(&mut **self)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

impl ser::SerializeStruct for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), OverLimit> {
        field(self, key, value)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

impl ser::SerializeStructVariant for &mut JsonLen {
    type Ok = ();
    type Error = OverLimit;
    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), OverLimit> {
        field(self, key, value)
    }
    fn end(self) -> Result<(), OverLimit> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_at_least_the_compact_json_of_its_unescaped_text() {
        let value = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": null}],
            "n": 1, "stream": false, "temperature": 0.5,
        });
        let measured = json_len_within(&value, usize::MAX).unwrap();
        assert!(measured >= serde_json::to_vec(&value).unwrap().len());
    }

    #[test]
    fn stops_past_the_limit_without_scanning_long_strings() {
        let value = serde_json::json!({"input": "x".repeat(1 << 20)});
        assert_eq!(json_len_within(&value, 16 * 1024), None);
        assert_eq!(json_len_within(&value, usize::MAX), Some((1 << 20) + 13));
    }
}
