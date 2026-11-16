//! Serialize a value with every string clipped, without an intermediate
//! `serde_json::Value`.
//!
//! Going through `Value` would widen every `f32` to `f64` before it is
//! written (`0.9` reaches the wire as `0.8999999761581421`), so the clip is
//! applied while serializing instead: each string the value hands to the
//! serializer is cut, everything else is forwarded unchanged, and the bytes
//! are those the value serializes to directly. Map keys are not clipped.

use serde::ser::{
    Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant, Serializer,
};

/// `T`, serialized with each string clipped to `max_bytes` (backing off to
/// a character boundary).
pub(super) struct Clipped<'a, T: ?Sized> {
    pub value: &'a T,
    pub max_bytes: usize,
}

impl<T: Serialize + ?Sized> Serialize for Clipped<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.value.serialize(Clip {
            inner: serializer,
            max_bytes: self.max_bytes,
        })
    }
}

struct Clip<S> {
    inner: S,
    max_bytes: usize,
}

impl<S> Clip<S> {
    fn of<'a, T: ?Sized>(&self, value: &'a T) -> Clipped<'a, T> {
        Clipped {
            value,
            max_bytes: self.max_bytes,
        }
    }
}

macro_rules! forward {
    ($($method:ident($ty:ty)),* $(,)?) => {
        $(fn $method(self, v: $ty) -> Result<S::Ok, S::Error> {
            self.inner.$method(v)
        })*
    };
}

impl<S: Serializer> Serializer for Clip<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = Clip<S::SerializeSeq>;
    type SerializeTuple = Clip<S::SerializeTuple>;
    type SerializeTupleStruct = Clip<S::SerializeTupleStruct>;
    type SerializeTupleVariant = Clip<S::SerializeTupleVariant>;
    type SerializeMap = Clip<S::SerializeMap>;
    type SerializeStruct = Clip<S::SerializeStruct>;
    type SerializeStructVariant = Clip<S::SerializeStructVariant>;

    forward!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_i128(i128),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_u128(u128),
        serialize_f32(f32),
        serialize_f64(f64),
        serialize_char(char),
        serialize_bytes(&[u8]),
        serialize_unit_struct(&'static str),
    );

    fn serialize_str(self, v: &str) -> Result<S::Ok, S::Error> {
        self.inner
            .serialize_str(&v[..v.floor_char_boundary(self.max_bytes)])
    }

    fn serialize_none(self) -> Result<S::Ok, S::Error> {
        self.inner.serialize_none()
    }

    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<S::Ok, S::Error> {
        let value = self.of(value);
        self.inner.serialize_some(&value)
    }

    fn serialize_unit(self) -> Result<S::Ok, S::Error> {
        self.inner.serialize_unit()
    }

    fn serialize_unit_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
    ) -> Result<S::Ok, S::Error> {
        self.inner.serialize_unit_variant(name, index, variant)
    }

    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<S::Ok, S::Error> {
        let value = self.of(value);
        self.inner.serialize_newtype_struct(name, &value)
    }

    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<S::Ok, S::Error> {
        let value = self.of(value);
        self.inner
            .serialize_newtype_variant(name, index, variant, &value)
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, S::Error> {
        let inner = self.inner.serialize_seq(len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, S::Error> {
        let inner = self.inner.serialize_tuple(len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, S::Error> {
        let inner = self.inner.serialize_tuple_struct(name, len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, S::Error> {
        let inner = self
            .inner
            .serialize_tuple_variant(name, index, variant, len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, S::Error> {
        let inner = self.inner.serialize_map(len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, S::Error> {
        let inner = self.inner.serialize_struct(name, len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, S::Error> {
        let inner = self
            .inner
            .serialize_struct_variant(name, index, variant, len)?;
        Ok(Clip {
            inner,
            max_bytes: self.max_bytes,
        })
    }

    fn is_human_readable(&self) -> bool {
        self.inner.is_human_readable()
    }
}

macro_rules! clip_elements {
    ($($trait:ident :: $method:ident),* $(,)?) => {
        $(impl<C: $trait> $trait for Clip<C> {
            type Ok = C::Ok;
            type Error = C::Error;

            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), C::Error> {
                let value = self.of(value);
                self.inner.$method(&value)
            }

            fn end(self) -> Result<C::Ok, C::Error> {
                self.inner.end()
            }
        })*
    };
}

clip_elements!(
    SerializeSeq::serialize_element,
    SerializeTuple::serialize_element,
    SerializeTupleStruct::serialize_field,
    SerializeTupleVariant::serialize_field,
);

macro_rules! clip_fields {
    ($($trait:ident),* $(,)?) => {
        $(impl<C: $trait> $trait for Clip<C> {
            type Ok = C::Ok;
            type Error = C::Error;

            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> Result<(), C::Error> {
                let value = self.of(value);
                self.inner.serialize_field(key, &value)
            }

            fn skip_field(&mut self, key: &'static str) -> Result<(), C::Error> {
                self.inner.skip_field(key)
            }

            fn end(self) -> Result<C::Ok, C::Error> {
                self.inner.end()
            }
        })*
    };
}

clip_fields!(SerializeStruct, SerializeStructVariant);

impl<C: SerializeMap> SerializeMap for Clip<C> {
    type Ok = C::Ok;
    type Error = C::Error;

    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), C::Error> {
        self.inner.serialize_key(key)
    }

    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), C::Error> {
        let value = self.of(value);
        self.inner.serialize_value(&value)
    }

    fn serialize_entry<K: Serialize + ?Sized, V: Serialize + ?Sized>(
        &mut self,
        key: &K,
        value: &V,
    ) -> Result<(), C::Error> {
        let value = self.of(value);
        self.inner.serialize_entry(key, &value)
    }

    fn end(self) -> Result<C::Ok, C::Error> {
        self.inner.end()
    }
}
