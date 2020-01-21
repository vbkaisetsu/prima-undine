extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;
use syn::Meta;
use syn::NestedMeta;

#[proc_macro_derive(Model, attributes(parameter))]
pub fn derive_model(input: TokenStream) -> TokenStream {
    let ast: DeriveInput = syn::parse(input).unwrap();
    let name = ast.ident;
    let generics = ast.generics;
    let gparams = &generics.params;
    let gwhere = generics.where_clause;
    let mut gtypes = vec![];
    let mut glifetimes = vec![];
    let mut gconsts = vec![];
    for param in generics.params.iter() {
        match param {
            syn::GenericParam::Type(x) => {
                gtypes.push(&x.ident);
            }
            syn::GenericParam::Lifetime(x) => {
                glifetimes.push(&x.lifetime);
            }
            syn::GenericParam::Const(x) => {
                gconsts.push(&x.ident);
            }
        }
    }
    if let Data::Struct(data) = ast.data {
        if let Fields::Named(fields) = data.fields {
            let mut field_updater = vec![];
            'a: for field in fields.named.iter() {
                for attr in &field.attrs {
                    if let Ok(meta) = attr.parse_meta() {
                        if let Meta::List(lst) = meta {
                            if lst.path.is_ident("parameter") {
                                for child in lst.nested.iter() {
                                    if let NestedMeta::Meta(Meta::Path(meta)) = child {
                                        if meta.is_ident("skip") {
                                            continue 'a;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                field_updater.push(field.ident.as_ref().unwrap());
            }
            let first_lifetime = glifetimes[0];
            let tokens = quote! {
                impl<#gparams> prima_undine::Model<#first_lifetime> for #name<#(#glifetimes ,)* #(#gtypes ,)* #({#gconsts} ,)*> #gwhere {
                    fn parameters(&self) -> Vec<&prima_undine::Parameter<#first_lifetime>> {
                        let mut params = vec![];
                        #( params.append(&mut self.#field_updater.parameters()); )*
                        params
                    }
                    fn parameters_mut(&mut self) -> Vec<&mut prima_undine::Parameter<#first_lifetime>> {
                        let mut params = vec![];
                        #( params.append(&mut self.#field_updater.parameters_mut()); )*
                        params
                    }
                }
            };
            tokens.into()
        } else {
            panic!("Model derive only supports a named struct");
        }
    } else {
        panic!("Model derive only supports a named struct");
    }
}
