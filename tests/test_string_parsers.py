import pytest
from slice.string_parsers import parse_entity_names, remove_special_characters, clean_entity

class TestParseEntityNames:
    """Test suite for parse_entity_names function"""
    
    def test_basic_parsing(self):
        """Test basic parsing with default delimiters"""
        result = parse_entity_names("col1, col2; col3")
        assert result == {"col1", "col2", "col3"}
        
    def test_single_name(self):
        """Test parsing a single name without delimiters"""
        result = parse_entity_names("single_name")
        assert result == {"single_name"}
        
    def test_empty_string(self):
        """Test handling of empty string input"""
        result = parse_entity_names("")
        assert result == set()
        
    def test_whitespace_only(self):
        """Test handling of whitespace-only input"""
        result = parse_entity_names("   ")
        assert result == set()
        
    def test_custom_delimiters(self):
        """Test parsing with custom delimiters and whitespace"""
        result = parse_entity_names("""
        MODULE @"/shares/IDEAs.Prod.Data/Publish.Profiles.Tenant.Commercial.IDEAsTenantProfile/Resources/v4/IDEAsTenantProfileExtension_v2.module" AS IDEAsTenantProfileExtension ; IDEAsTenantProfileExtension.IDEAsTenantProfileExtensionView(Extensions=new ARRAY<string>{
        "IDEAsAvailableUnits","IDEAsCloudAscentData","IDEAsDomains","IDEAsExternalEnableUsers","IDEAsExternalUsers","IDEAsFirstPaidDates","IDEAsInternal","IDEAsMSSales","IDEAsPublicSector","IDEAsSKU","IDEAsSubscription","IDEAsTenantTags","IDEAsViral","IDEAsFastTrackTenants","IDEAsCALC","IDEAsHasWorkloads"})
        """, delimiters=[";"])
        assert result == {
            "IDEAsTenantProfileExtension_IDEAsTenantProfileExtensionView_Extensions_new_ARRAY_string_IDEAsAvailableUnits_IDEAsCloudAscentData_IDEAsDomains_IDEAsExternalEnableUsers_IDEAsExternalUsers_IDEAsFirstPaidDates_IDEAsInternal_IDEAsMSSales_IDEAsPublicSector_IDEAsSKU_IDEAsSubscription_IDEAsTenantTags_IDEAsViral_IDEAsFastTrackTenants_IDEAsCALC_IDEAsHasWorkloads",
            "MODULE_shares_IDEAs_Prod_Data_Publish_Profiles_Tenant_Commercial_IDEAsTenantProfile_Resources_v4_IDEAsTenantProfileExtension_v2_module_AS_IDEAsTenantProfileExtension"
        }
        
    def test_mixed_delimiters(self):
        """Test parsing with mixed delimiters"""
        result = parse_entity_names("col1,col2;col3|col4", delimiters=[",", ";", "|"])
        assert result == {"col1", "col2", "col3", "col4"}
        
    def test_whitespace_handling(self):
        """Test proper handling of whitespace around delimiters"""
        result = parse_entity_names("  col1  ,  col2  ;  col3  ")
        assert result == {"col1", "col2", "col3"}

class TestRemoveSpecialCharacters:
    """Test suite for remove_special_characters function"""
    
    def test_basic_removal(self):
        """Test basic character removal with default replacement"""
        result = remove_special_characters("""MODULE @"/shares/IDEAs.Prod.Data/Publish.Profiles.Tenant.Commercial.IDEAsTenantProfile/Resources/v4/IDEAsTenantProfileExtension_v2.module" AS IDEAsTenantProfileExtension""", ["@", '"', ' '])
        assert result == "MODULE_/shares/IDEAs.Prod.Data/Publish.Profiles.Tenant.Commercial.IDEAsTenantProfile/Resources/v4/IDEAsTenantProfileExtension_v2.module_AS_IDEAsTenantProfileExtension"
        
    def test_custom_replacement(self):
        """Test character removal with custom replacement"""
        result = remove_special_characters("test!@#string", ["!", "@", "#"], replace_char="-")
        assert result == "test-string"
        
    def test_no_characters_to_remove(self):
        """Test when no characters need to be removed"""
        result = remove_special_characters("teststring", [], replace_char="_")
        assert result == "teststring"
        
    def test_empty_string(self):
        """Test handling of empty string input"""
        result = remove_special_characters("", ["!", "@", "#"], replace_char="_")
        assert result == ""
        
    def test_special_characters_only(self):
        """Test when string contains only special characters"""
        result = remove_special_characters("!@#", ["!", "@", "#"], replace_char="_")
        assert result == "_"
        
    def test_multiple_occurrences(self):
        """Test removal of multiple occurrences of special characters"""
        result = remove_special_characters("test!!string@@test", ["!", "@"], replace_char="_")
        assert result == "test_string_test"
        
    def test_whitespace_handling(self):
        """Test proper handling of whitespace"""
        result = remove_special_characters("test ! string", ["!"], replace_char="_")
        assert result == "test _ string"

class TestCleanEntity:
    """Test suite for clean_entity function"""
    
    def test_default_behavior(self):
        """Test default behavior (keeping letters, numbers, underscores)"""
        result = clean_entity("Hello World!")
        assert result == "Hello_World"
        
    def test_custom_replacement(self):
        """Test custom replacement character"""
        result = clean_entity("Hello World!", replace_char="-")
        assert result == "Hello-World"
        
    def test_custom_chars_removal(self):
        """Test custom character removal"""
        result = clean_entity("Hello@World#Test", chars_to_remove=["@", "#"])
        assert result == "Hello_World_Test"
        
    def test_empty_string(self):
        """Test handling of empty string"""
        result = clean_entity("")
        assert result == ""
        
    def test_already_clean(self):
        """Test string that's already clean"""
        result = clean_entity("abc_123")
        assert result == "abc_123"
        
    def test_only_special_chars(self):
        """Test string with only special characters"""
        result = clean_entity("!!!")
        assert result == ""
        
    def test_whitespace(self):
        """Test handling of whitespace"""
        result = clean_entity("  spaces  ")
        assert result == "spaces"
        
    def test_consecutive_special_chars(self):
        """Test handling of consecutive special characters"""
        result = clean_entity("hello!!!world")
        assert result == "hello_world"
        
    def test_complex_path(self):
        """Test cleaning complex path string"""
        result = clean_entity("""/path/to/file@name.ext""")
        assert result == "path_to_file_name_ext" 