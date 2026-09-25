#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace local_gguf_engine::tests {

// Independent CPU oracle, scripts/gemma3_reference_golden.py: authenticated
// real Q8 checkpoint, PyTorch 2.14.0 / NumPy 2.5.2, CPU F32 with F32 cached K/V.
// The rejected F16-cache oracle remains separately preserved; no tolerance was
// relaxed. These hashes identify build/gemma3-reference-f32-golden, schema v2.
// Binary outputs stay ignored and cannot be replaced by self-generated Metal
// output or accidentally stale reference evidence.
struct Gemma3GoldenDigest {
    std::string_view filename;
    std::string_view sha256;
    std::size_t rows;
};
inline constexpr std::string_view gemma3_golden_identity_sha256 = "888b2ceab5ce29e8c3e5deac6c9b06d5fac4503de94553c568f3d3aca83d794f";
inline constexpr std::array<Gemma3GoldenDigest, 77> gemma3_golden_digests = {{
    {"decode-layer-00-key.f32", "a1bf0acf249d7bd7292bc8fedfb3510b81fc7c3ca6a60ffbbf1541dddb5d9e3f", 4},
    {"decode-layer-00-value.f32", "7b4276c1aef78be76e2b5b88d7da0d6b0f5f703602c9ac8d7bc094220c1e06de", 4},
    {"decode-layer-01-key.f32", "98be54ddf8b0b3f952aa844d3fa4325f33096dc88fd347ca47d191ee8d5a9152", 4},
    {"decode-layer-01-value.f32", "85e59c7baecc9b5bdd9b08660f18855a29fac8a2eba938a30d5eaa6c8c62b5b5", 4},
    {"decode-layer-02-key.f32", "c59b1a6c769b0094762c6d0e1a00fe02df517425f933fb035ce5decb90f53255", 4},
    {"decode-layer-02-value.f32", "06b004af17a5a6717ad1564dd878900ac1c2931925cdf3a413ea6fe916686d7e", 4},
    {"decode-layer-03-key.f32", "38f2374a974bb8ff93d88913c3afe373e4b791af59a48709d34b5fc70b9676ae", 4},
    {"decode-layer-03-value.f32", "a5fe77b002d09720d3ad39606c149e747875dcd5172997d122bcfb5d70c2c37c", 4},
    {"decode-layer-04-key.f32", "7c336aa3bd058b7636b0f0a232597038edac8f0030dfdacedb64126da3fb6c49", 4},
    {"decode-layer-04-value.f32", "41b9b36b88a03992cae3f295ae925c5d6087950e17ed64799a61aac5b0158416", 4},
    {"decode-layer-05-key.f32", "9af5cb8de48d32ee3829b2ffcb750c685b0075a65c4fa8c8012267a2094caa51", 4},
    {"decode-layer-05-value.f32", "c08e37acbe1b5196abbabde5809643b09ba0084e98c094f5fbc00ac04d292db7", 4},
    {"decode-layer-06-key.f32", "65a5521ae5090d81948e0b13952e9e40fc406a09c67ef3ee4d8ab86b98ee2c9c", 4},
    {"decode-layer-06-value.f32", "1db98216af44f7f0fc4cdc80644475f8bdb93def5dda4efe50b1f1117b6b8012", 4},
    {"decode-layer-07-key.f32", "f573c157bf1bfe89ec204ec390ff50cd5f3c105e92fe3ab0e74404886030661e", 4},
    {"decode-layer-07-value.f32", "6279bf2c9855d77bea2b4fa8479f32b8aa446243fe8799a4f48ba51fcdc8f0cc", 4},
    {"decode-layer-08-key.f32", "45b331e26f7e012060a4e96a8758c0d7e893966c8e90f5ff69508a2ca27909d9", 4},
    {"decode-layer-08-value.f32", "898fb444c028713b6379e484c428b0ec79da44d18f978ddef1586b405923d91d", 4},
    {"decode-layer-09-key.f32", "af6883dcbc69b23a8aacd64b5a039ad9dd27ccdb3f2341274c234e4631137b03", 4},
    {"decode-layer-09-value.f32", "82d4bccca2e90f9d5563e8c2547c7b813bc10f6a652388e42f28ee5eed7ba1aa", 4},
    {"decode-layer-10-key.f32", "b78a21576c53ce52217ba98edd30fd7895d980c35a990636cc5fa2648016ddbe", 4},
    {"decode-layer-10-value.f32", "a0a61961e1c1ccd228299b37099c8491ff2e916ded244a5b2735cba2a30ee3a4", 4},
    {"decode-layer-11-key.f32", "003c86347413cd47cff70e86c5ad96656d4a8238cb99ccd314b60f525e027097", 4},
    {"decode-layer-11-value.f32", "fc8096269e386a71f0ea8cfea506f11d9d662dc6c5648dd58d88b77f4fc44284", 4},
    {"decode-layer-12-key.f32", "8fc1fc9a64c0af2d62f2ec3091265229bc0c5495ff8afbded38ed99d12948577", 4},
    {"decode-layer-12-value.f32", "f8f7788cf64336d38f6f4241e549e190312c1d4a50501c3056b0c1a23f8c92cf", 4},
    {"decode-layer-13-key.f32", "9a79556dd8d6b92c6b0c57fcf9e864a9b4ae3c27a161cb62f2bd1ad297dd349c", 4},
    {"decode-layer-13-value.f32", "f96bb0e12dabcc6eaa96e6d197ffd2f9923a2e84fbb5adf0d83881d648ef7fcd", 4},
    {"decode-layer-14-key.f32", "cd946eff930a54712ed4933b6026e3200d668c94fc1c58b5dd9e3ea6497ac046", 4},
    {"decode-layer-14-value.f32", "b3acd0f35b5d369fb6ab6ca628441d655fb2bf02bdb05a969279cccd61402310", 4},
    {"decode-layer-15-key.f32", "e5ad0643d18db1712df580e6da5a58048b5d15f155e8cf65cf3eb8a9936853bd", 4},
    {"decode-layer-15-value.f32", "30bf7e70749154fd84c1378f1f02e7d6581d2bf0e596c8869d985ab224f872b5", 4},
    {"decode-layer-16-key.f32", "1a1224cf9fb550cd858a9e47a251370081015ea1e54e5e995122bbdd1a229941", 4},
    {"decode-layer-16-value.f32", "7f839f50bcc73b4bf6e6c18e0cf6c21f49dbb0ca83ee10f41c27c65bec986d4e", 4},
    {"decode-layer-17-key.f32", "8ba245f9fd98b2a4359caebf0c208a9d1c7cdf2db9507ea51ebb012886f5b6a5", 4},
    {"decode-layer-17-value.f32", "7c66d2eb15a82c3deda51e63deeb05366ba0268bbb0415c9a4be75faadeb9d18", 4},
    {"decode-logits.f32", "1170925dea87aed40a047c3131f725d660719545f620b0858acbe2d490ba3298", 262144},
    {"prefill-logits.f32", "bf2f1bfb28686992b5641e53f14abdd55e0073d5853d43b70f6bb365b79a63ae", 262144},
    {"wrap-after-layer-00-key.f32", "4fff5b54c5e4ce3fa98749bc2c520a9648c4aeb8a8913b2ce8e4dbaaa48cd4c4", 512},
    {"wrap-after-layer-00-value.f32", "ba1ee8392b0fff14f421aca6f8d2c5551e92c6bb54322ed9891b179e265dd8e0", 512},
    {"wrap-after-layer-01-key.f32", "20f9ee16aaea3580a658d8f4086c4cea4fe3a97b1c4d2b4402683a3f0bc954a9", 512},
    {"wrap-after-layer-01-value.f32", "20b51d393b0aa734372c86142ebfc5efda62144f830e10f1f6e95e41205111b6", 512},
    {"wrap-after-layer-02-key.f32", "7f03d11816a6e8c9a1533f9d179123cf0fabef8db078b155ffd2cb48b60460cc", 512},
    {"wrap-after-layer-02-value.f32", "c11253bf417706411c4d8bea38f91494f98c419504a06e9f8916fdd6cba78f2c", 512},
    {"wrap-after-layer-03-key.f32", "5320b892910512732a976defca94b306b42bc1bc8479ecfe74031e56233039fa", 512},
    {"wrap-after-layer-03-value.f32", "fb2f47d11dc918e0c8dee36db62b3eb82d9409d39ae47ad477756719276c4b82", 512},
    {"wrap-after-layer-04-key.f32", "13f5eba272b5137179907b544af3958c0f5d60c4a21bb2b582cd9f0645669d7e", 512},
    {"wrap-after-layer-04-value.f32", "5b7e060fdc1d01670c25d28bdfe9c02318bf99921f5b09842601e20eb2f1ea0a", 512},
    {"wrap-after-layer-05-key.f32", "47a0f3562c6bce19f6212eea261086c33aabada22f024233f27a54910db427c4", 536},
    {"wrap-after-layer-05-value.f32", "fc6554cbae1d94fcd81b3e613f3ce7dd4ceaacac67fd348f643f3d9c7ba25f49", 536},
    {"wrap-after-layer-06-key.f32", "9623976bd778a665058ec7d6d967cb63ddb3f070deb90f7fb7bccc67c1bf9636", 512},
    {"wrap-after-layer-06-value.f32", "a5e3c2e3daef167ead20af8ae555599e4c587c8baa11c5b3bf4660bcc5763498", 512},
    {"wrap-after-layer-07-key.f32", "306db1ac0e46fddc3ef11107278c93c296c0314bbcc46760ed54250828c87695", 512},
    {"wrap-after-layer-07-value.f32", "9a68cfd7160cb8fda5a8b8119b44b870bf203c2f45d5b786f4de079b7d83850d", 512},
    {"wrap-after-layer-08-key.f32", "a2a6019257c8d91f68c94d3f419c69dbf3e7f6c0264264a67c2e0270eafb429c", 512},
    {"wrap-after-layer-08-value.f32", "0012d8985af943919f6a5264ea52d91693498b73b9be7703f213f955f8fac354", 512},
    {"wrap-after-layer-09-key.f32", "36ac33ce1ceaf27745b86ade99aa69875610383083f077667c63ad1a724595e8", 512},
    {"wrap-after-layer-09-value.f32", "b7db600ac0724395c46bce5e652233ad9bdbfef9e603549be0758ee60f791492", 512},
    {"wrap-after-layer-10-key.f32", "158e56a194510fa763271d3c67860e5ccb1dfa972872ca9e1fe680dd8e5fc882", 512},
    {"wrap-after-layer-10-value.f32", "7d30b6022d03ab7fafbc3c5083b336ba66b7441197444365c39336cab48b808c", 512},
    {"wrap-after-layer-11-key.f32", "3b92421b29559b11d541aa8cdb55ed28b141de1985392d23fe297b350809ab86", 536},
    {"wrap-after-layer-11-value.f32", "49d48f65da30e1f6c3558ba011f6a21a41e28db1bc3202cbda518aac86640948", 536},
    {"wrap-after-layer-12-key.f32", "6c6be271b941d98fb5aa1c0bc68e0d83b67a819275946a015c08521b0184b15a", 512},
    {"wrap-after-layer-12-value.f32", "bcd383c676d07356ffa4bdf1cff848f4f9ad0d6b16e3e47b9db155f3d531f2df", 512},
    {"wrap-after-layer-13-key.f32", "1f9127b54ce48b203cd56585754c51957203b6a6256a84cf9ec90073e1a1a03b", 512},
    {"wrap-after-layer-13-value.f32", "5c7d443305387feffacd40426e2fa9168895908edd8f93dd98e24896c91d98a9", 512},
    {"wrap-after-layer-14-key.f32", "3c0347700d0965d0c82035cbc744ed5ae6d5740be0633e6c09b89e93733e20ab", 512},
    {"wrap-after-layer-14-value.f32", "6db44cadfc01e1770de6bc5138c3459ef149ace2d0107dba85a409978f417dba", 512},
    {"wrap-after-layer-15-key.f32", "4a771406b5bf259022cd675d232be42492d4e7861f067f078f6e5818a23e7669", 512},
    {"wrap-after-layer-15-value.f32", "f965851b76d4c2484927b7c127b0607e9f5386d76eeb70eabaa9a8ee5a791f8d", 512},
    {"wrap-after-layer-16-key.f32", "b68a966cf3fe1f321bdd5031a6cb06cb250c6e92eb7eae4e8ec3885c8076bd63", 512},
    {"wrap-after-layer-16-value.f32", "d753f784ae5ef6cc8299baea5dfea0a006ccb82115b8956e0c26a9b06df53a06", 512},
    {"wrap-after-layer-17-key.f32", "b815b9270b4da308d29fe2b577e7e6f0363a0ef7fe9abaf8c015d54c2bd6b0eb", 536},
    {"wrap-after-layer-17-value.f32", "a64cf5a6962969e88f1ac644cb421fbb1b7950de21a6363a00e461e38c871aaa", 536},
    {"wrap-after-logits.f32", "32bf9a567038ee0d55b62518d54669577daff06e8bd2dbc8fbf3e19643bb847d", 262144},
    {"wrap-before-logits.f32", "0d978ed28383866c0efffa0d246dd8ef627309dce116037aa7a6b4900ea42a5a", 262144},
    {"wrap-crossing-logits.f32", "3a0180ccbcda4250faa823a12c10fafc638aa797acbbb56e6eec5bf1bce1e4c6", 262144},
}};

} // namespace local_gguf_engine::tests
