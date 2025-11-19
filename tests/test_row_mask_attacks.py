import pytest
from reconstruction_tests.row_mask_attacks import attack_loop

def test_attack_loop_perfect_reconstruction():
    """Test that attack_loop achieves perfect accuracy with no noise."""
    results = attack_loop(
        nrows=100,
        nunique=2,
        mask_fraction=0.5,
        noise=0,
        max_samples=1000,
        batch_size=100,
        target_accuracy=1.0
    )
    
    # Should have at least one result
    assert len(results) > 0
    
    # Final accuracy should be 1.0
    final_accuracy = results[-1]['measure']
    assert final_accuracy == 1.0

def test_attack_loop_returns_results():
    """Test that attack_loop returns properly formatted results."""
    results = attack_loop(
        nrows=50,
        nunique=2,
        mask_fraction=0.5,
        noise=1,
        max_samples=500,
        batch_size=50
    )
    
    # Should have results
    assert len(results) > 0
    
    # Each result should have correct keys
    for result in results:
        assert 'num_samples' in result
        assert 'measure' in result
        assert isinstance(result['num_samples'], int)
        assert isinstance(result['measure'], float)
        assert 0.0 <= result['measure'] <= 1.0

def test_attack_loop_stops_at_target():
    """Test that attack_loop stops when target accuracy is reached."""
    results = attack_loop(
        nrows=100,
        nunique=2,
        mask_fraction=0.5,
        noise=0,
        max_samples=10000,
        batch_size=100,
        target_accuracy=0.99
    )
    
    # Should stop before max_samples
    final_num_samples = results[-1]['num_samples']
    assert final_num_samples < 10000
    
    # Final accuracy should be >= target
    final_accuracy = results[-1]['measure']
    assert final_accuracy >= 0.99

def test_attack_loop_respects_max_samples():
    """Test that attack_loop respects max_samples limit."""
    results = attack_loop(
        nrows=100,
        nunique=3,
        mask_fraction=0.3,
        noise=5,
        max_samples=300,
        batch_size=100,
        target_accuracy=1.0  # Unlikely to reach with high noise
    )
    
    # Should not exceed max_samples
    final_num_samples = results[-1]['num_samples']
    assert final_num_samples <= 300
    final_num_samples = results[-1]['num_samples']
    assert final_num_samples <= 300
