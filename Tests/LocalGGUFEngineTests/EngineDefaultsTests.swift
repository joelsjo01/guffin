import Testing
@testable import LocalGGUFEngine

@Test func defaultBackendStartsOnCPU() {
    #expect(EngineDefaults.backend == .cpu)
}
